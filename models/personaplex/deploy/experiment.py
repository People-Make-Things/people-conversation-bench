"""Modal offline proof for PersonaPlex duplex teacher forcing."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import modal

from common import HF_SECRET_NAME, PERSONAPLEX_COMMIT, artifacts_vol, hf_cache_vol, image

app = modal.App("people-bench-personaplex-prefill-proof")


@app.function(
    image=image,
    gpu="A10G",
    timeout=30 * 60,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/artifacts": artifacts_vol,
    },
)
def run_prefill_proof(
    user_wav_b64: str | None = None,
    assistant_wav_b64: str | None = None,
    words_json: str | None = None,
) -> dict:
    import base64
    import tempfile
    import time

    import numpy as np
    import sentencepiece
    import sphn
    import torch
    from huggingface_hub import hf_hub_download

    from moshi.models import LMGen, loaders

    sys.path[:0] = ["/root/personaplex_server", "/root"]
    from prefill import run_duplex_prefill

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for the prefill proof")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_repo = loaders.DEFAULT_REPO
    hf_hub_download(hf_repo, "config.json")

    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)

    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    lm = loaders.get_moshi_lm(moshi_weight, device=device)
    lm.eval()

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=False,
    )
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    if user_wav_b64 and assistant_wav_b64:
        with tempfile.TemporaryDirectory() as tmp:
            user_path = Path(tmp) / "user.wav"
            assistant_path = Path(tmp) / "assistant.wav"
            user_path.write_bytes(base64.b64decode(user_wav_b64))
            assistant_path.write_bytes(base64.b64decode(assistant_wav_b64))
            user_pcm = sphn.read(user_path)[0][0].astype(np.float32)
            assistant_pcm = sphn.read(assistant_path)[0][0].astype(np.float32)
        words = json.loads(words_json or '{"words": []}').get("words", [])
    else:
        seconds = 1.6
        sample_count = int(seconds * mimi.sample_rate)
        sample_count -= sample_count % frame_size
        user_pcm = np.zeros(sample_count, dtype=np.float32)
        assistant_pcm = np.zeros(sample_count, dtype=np.float32)
        words = []

    mimi.reset_streaming()
    other_mimi.reset_streaming()
    lm_gen.reset_streaming()

    prefill_diag = asyncio.run(
        run_duplex_prefill(
            mimi,
            other_mimi,
            lm_gen,
            text_tokenizer,
            user_pcm,
            assistant_pcm,
            words,
            device,
            frame_size,
        )
    )

    live_pcm = np.zeros(frame_size, dtype=np.float32)
    live_chunk = torch.from_numpy(live_pcm).to(device=device)[None, None]
    live_codes = mimi.encode(live_chunk)
    started = time.monotonic()
    live_tokens = lm_gen.step(live_codes)
    live_elapsed = time.monotonic() - started

    result = {
        "prefill": {
            "frame_count": prefill_diag.frame_count,
            "history_duration_sec": prefill_diag.history_duration_sec,
            "elapsed_sec": prefill_diag.elapsed_sec,
            "realtime_factor": prefill_diag.realtime_factor,
        },
        "live_step_ok": live_tokens is not None,
        "live_step_elapsed_sec": live_elapsed,
        "personaplex_commit": PERSONAPLEX_COMMIT,
    }

    out_path = Path("/root/artifacts/prefill_proof.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    artifacts_vol.commit()
    return result


@app.local_entrypoint()
def main():
    result = run_prefill_proof.remote()
    print(json.dumps(result, indent=2))
