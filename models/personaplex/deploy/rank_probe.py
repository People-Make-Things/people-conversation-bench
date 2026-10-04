"""Modal GPU probe: rank of teacher-forced text tokens during prefill."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import modal

from common import HF_SECRET_NAME, PERSONAPLEX_COMMIT, artifacts_vol, hf_cache_vol, image

app = modal.App("people-bench-personaplex-rank-probe")

PROBE_DIR = "rank_probe"
PROBE_SEED = 424242
PROBE_POINTS = "000006,000016,000029"
CONTINUATION_SEC = 15.0


@app.function(
    image=image,
    gpu="A10G",
    timeout=3 * 60 * 60,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/artifacts": artifacts_vol,
    },
)
def run_rank_probe(
    histories: list[dict],
    voice_prompts: list[str | None],
    continuation_sec: float = CONTINUATION_SEC,
    seeds: int = 1,
    label: str = PROBE_DIR,
) -> dict:
    """Rank the model gives the text token prefill forces, per prefill condition.

    Runs the real `run_duplex_prefill` behind the real `ServerState`, so the only
    things varying are the voice prompt and the history each arm is primed with.
    Every arm then continues on silence frames the way the drain does, which is
    what turns a rank trajectory into a time-to-first-text. Prefill is fully
    teacher-forced and so deterministic; the seeds vary the continuation, which
    is where the outcome is measured.
    """
    import numpy as np
    import sentencepiece
    import sphn
    import torch
    from huggingface_hub import hf_hub_download

    from moshi.models import loaders

    sys.path[:0] = ["/root/personaplex_server", "/root"]
    from duplex_server import ServerState, _get_voice_prompt_dir, seed_all
    from prefill import encode_frame, run_duplex_prefill
    from text_alignment import EPAD_TOKEN, PAD_TOKEN
    from bench.speech import SPEECH_RMS_THRESHOLD

    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for the rank probe")

    # duplex_server.main runs under `with torch.no_grad()`, and the mimi passes
    # in prefill inherit that from their caller rather than declaring it, so
    # without this the A10G runs out of memory on autograd state alone.
    torch.set_grad_enabled(False)

    # A reused container holds the volume as it was at start, which predates the
    # arms this call just uploaded.
    artifacts_vol.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_repo = loaders.DEFAULT_REPO
    hf_hub_download(hf_repo, "config.json")
    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
    lm = loaders.get_moshi_lm(moshi_weight, device=device)
    lm.eval()

    state = ServerState(
        loaders.get_mimi(mimi_weight, device),
        loaders.get_mimi(mimi_weight, device),
        text_tokenizer,
        lm,
        device,
        voice_prompt_dir=_get_voice_prompt_dir(None, hf_repo),
    )
    state.warmup()

    class ForcedTextRank:
        """Rank of the forced text token at each prefill step.

        The same quantity `create_loss_report` writes to `ranks_of_forced[:, 0]`,
        taken off the text logits as they are produced instead, by wrapping the
        graphed forward. `prepare_step_input` has already written the forced
        token into the cache at `offset + delays[0]` by then, and text sits at
        delay 0, so the token at `offset` is the one these logits predict.

        LMGen(report_loss=True) cannot supply this on the pinned commit: it
        forces return_logits on and then `create_loss_report` rebinds `target`
        to the text channel at lm.py:601 before indexing `target[:, k + 1]` at
        lm.py:623, which raises IndexError on the now 1-D tensor for any
        dep_q > 0. The logits are copied to the host because the A10G runs this
        model with tens of megabytes spare and a per-step device allocation
        fragments the pool into an OOM.
        """

        def __init__(self, lm_gen):
            self.lm_gen = lm_gen
            self.inner = lm_gen._streaming_state.graphed_main
            lm_gen._streaming_state.graphed_main = self
            self.host: torch.Tensor | None = None
            self.forced: list[int] = []
            self.rank: list[int] = []
            self.prob: list[float] = []
            self.top: list[int] = []
            self.top_prob: list[float] = []

        def __call__(self, input_):
            transformer_out, text_logits = self.inner(input_)
            state = self.lm_gen._streaming_state
            position = state.offset % state.cache.shape[2]
            forced = int(state.cache[0, 0, position].item())
            source = text_logits[0, 0, 0]
            if self.host is None:
                self.host = torch.empty(source.shape, dtype=source.dtype, device="cpu")
            self.host.copy_(source)
            logits = self.host.float()
            probs = torch.softmax(logits, dim=-1)
            top = int(torch.argmax(logits).item())
            self.forced.append(forced)
            self.rank.append(int((logits > logits[forced]).sum().item()))
            self.prob.append(float(probs[forced].item()))
            self.top.append(top)
            self.top_prob.append(float(probs[top].item()))
            return transformer_out, text_logits

        def restore(self) -> None:
            self.lm_gen._streaming_state.graphed_main = self.inner

    def continue_on_silence(frames: int) -> dict:
        """Step the model on silence, as the pause drain does, and watch its text."""
        silence = np.zeros(state.frame_size, dtype=np.float32)
        pieces: list[str] = []
        first_text_frame: int | None = None
        gated = 0
        gated_before_text = 0
        for frame_index in range(frames):
            codes = encode_frame(state.mimi, silence, device)
            tokens = state.lm_gen.step(codes)
            if tokens is None:
                continue
            audio = state.mimi.decode(tokens[:, 1:9])[0, 0].to("cpu")
            # Frame RMS against the scoring threshold, not the hysteresis gate.
            above_gate = bool(audio.float().square().mean().sqrt() >= SPEECH_RMS_THRESHOLD)
            gated += int(above_gate)
            text_token = int(tokens[0, 0, 0].item())
            if text_token not in (EPAD_TOKEN, PAD_TOKEN):
                if first_text_frame is None:
                    first_text_frame = frame_index
                pieces.append(text_tokenizer.id_to_piece(text_token).replace("\u2581", " "))
            elif first_text_frame is None:
                gated_before_text += int(above_gate)
        return {
            "frames": frames,
            "first_text_frame": first_text_frame,
            "text_tokens": len(pieces),
            "text": "".join(pieces),
            "gated_frames": gated,
            "gated_frames_before_text": gated_before_text,
        }

    arms = []
    report = {
        "personaplex_commit": PERSONAPLEX_COMMIT,
        "seed": PROBE_SEED,
        "seeds": seeds,
        "continuation_sec": continuation_sec,
        "arms": arms,
    }
    out_path = Path("/root/artifacts") / label / "report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Seeds outermost so a run that dies partway still leaves whole sweeps of
    # every arm on the volume rather than a prefix of the arms at one seed.
    for seed_index in range(seeds):
        for history in histories:
            directory = Path("/root/artifacts") / label / history["dir"]
            user_pcm = sphn.read(directory / "user.wav")[0][0].astype(np.float32)
            assistant_pcm = sphn.read(directory / "assistant.wav")[0][0].astype(np.float32)
            meta = json.loads((directory / "meta.json").read_text())
            for voice_prompt in voice_prompts:
                seed_all(PROBE_SEED + seed_index)
                state.mimi.reset_streaming()
                state.other_mimi.reset_streaming()
                state.lm_gen.reset_streaming()
                state.configure_session(
                    state.resolve_voice_prompt(voice_prompt), meta["text_prompt"]
                )
                state.lm_gen.step_system_prompts(state.mimi)
                state.mimi.reset_streaming()

                capture = ForcedTextRank(state.lm_gen)
                diagnostics = asyncio.run(
                    run_duplex_prefill(
                        state.mimi,
                        state.other_mimi,
                        state.lm_gen,
                        state.text_tokenizer,
                        user_pcm,
                        assistant_pcm,
                        meta["words"],
                        device,
                        state.frame_size,
                    )
                )
                capture.restore()
                arm = {
                    "point_id": meta["point_id"],
                    "history": history["history"],
                    "arm": history.get("arm", "control"),
                    "masked_sec": history.get("masked_sec", 0.0),
                    "history_sec": diagnostics.history_duration_sec,
                    "voice_prompt": voice_prompt,
                    "seed": PROBE_SEED + seed_index,
                    "frame_count": diagnostics.frame_count,
                    "realtime_factor": diagnostics.realtime_factor,
                    "forced": capture.forced,
                    "rank": capture.rank,
                    "prob": capture.prob,
                    "top": capture.top,
                    "top_prob": capture.top_prob,
                    "continuation": continue_on_silence(
                        int(continuation_sec * state.mimi.frame_rate)
                    ),
                }
                arms.append(arm)
                print(
                    f"{arm['point_id']} {arm['arm']} {arm['history']} "
                    f"voice={voice_prompt} seed={arm['seed']} "
                    f"frames={arm['frame_count']} "
                    f"first_text_frame={arm['continuation']['first_text_frame']}",
                    flush=True,
                )

        out_path.write_text(json.dumps(report))
        artifacts_vol.commit()
    return {
        "arms": [
            {
                key: arm[key]
                for key in ("point_id", "arm", "history", "history_sec", "seed", "frame_count")
            }
            | {"first_text_frame": arm["continuation"]["first_text_frame"]}
            for arm in arms
        ],
    }


def stage_histories(label: str, entries: list[dict]) -> list[dict]:
    """Upload each arm's duplex history to the artifacts volume.

    Function arguments cannot carry minutes of PCM, so the audio goes through the
    volume and the remote function reads it back from there.
    """
    import tempfile

    from bench.audio import write_wav_pcm16

    histories = []
    with tempfile.TemporaryDirectory() as staging:
        with artifacts_vol.batch_upload(force=True) as batch:
            for entry in entries:
                local = Path(staging) / entry["dir"]
                local.mkdir(parents=True)
                rate = entry["sample_rate"]
                write_wav_pcm16(local / "user.wav", entry["user_pcm"], rate)
                write_wav_pcm16(local / "assistant.wav", entry["assistant_pcm"], rate)
                (local / "meta.json").write_text(
                    json.dumps(
                        {
                            "point_id": entry["point_id"],
                            "text_prompt": entry["text_prompt"],
                            "words": entry["words"],
                        }
                    )
                )
                batch.put_directory(str(local), f"/{label}/{entry['dir']}")
                histories.append(
                    {
                        key: entry[key]
                        for key in ("dir", "history", "arm", "masked_sec")
                        if key in entry
                    }
                )
    return histories


def download_report(label: str, out: str) -> Path:
    destination = Path(out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        for chunk in artifacts_vol.read_file(f"{label}/report.json"):
            handle.write(chunk)
    return destination.resolve()


def timed_word_dicts(words) -> list[dict]:
    return [
        {"text": word.text, "start_sec": word.start_sec, "end_sec": word.end_sec}
        for word in words
    ]


@app.local_entrypoint()
def probe(
    points: str = PROBE_POINTS,
    manifest_dir: str = "data/processed/example_1/points",
    history_sec: float = 4.0,
    continuation_sec: float = CONTINUATION_SEC,
    seeds: int = 1,
    out: str = "data/analysis/voice_prompt/rank_probe.json",
):
    """Voice prompt x history-length factorial over the forced-text rank."""
    from bench.history import condition_duplex_history
    from bench.points import load_rollout_scenario
    from bench.protocol import ContextType

    entries = []
    for point in points.split(","):
        scenario = load_rollout_scenario(
            Path(manifest_dir) / point.strip(), kinds=(ContextType.DUPLEX_AUDIO,)
        )
        for name, seconds in (("full", None), (f"{history_sec:g}s", history_sec)):
            context = condition_duplex_history(scenario, history_sec=seconds).contexts[
                ContextType.DUPLEX_AUDIO
            ]
            entries.append(
                {
                    "dir": f"{scenario.point_id}_{name}",
                    "history": name,
                    "point_id": scenario.point_id,
                    "text_prompt": scenario.text_prompt,
                    "words": timed_word_dicts(context.words),
                    "user_pcm": context.user_pcm,
                    "assistant_pcm": context.assistant_pcm,
                    "sample_rate": context.sample_rate,
                }
            )

    summary = run_rank_probe.remote(
        histories=stage_histories(PROBE_DIR, entries),
        voice_prompts=["NATF2.pt", None],
        continuation_sec=continuation_sec,
        seeds=seeds,
        label=PROBE_DIR,
    )
    print(json.dumps(summary, indent=2))
    print(download_report(PROBE_DIR, out))

