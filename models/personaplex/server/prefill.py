"""Teacher-force synchronized duplex history through PersonaPlex LMGen."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from text_alignment import aligned_text_tensor
from utils.timed_words import parse_timed_words

logger = logging.getLogger(__name__)


# Official trained window, not the Moshi inference ring size. NVIDIA:
# "2048-token context window, corresponding to roughly 160 seconds of audio."
# https://arxiv.org/abs/2602.06053 (max sequence length 2048 = 163.84 s)
# https://huggingface.co/nvidia/personaplex-7b-v1/blob/main/explainability.md
# quoted in https://github.com/NVIDIA/personaplex/issues/3
# Keep this equal to bench.audio.PERSONAPLEX_MAX_HISTORY_FRAMES. This server
# image does not import bench.
PERSONAPLEX_MAX_HISTORY_FRAMES = 2048

# Kyutai Moshi temporal transformer context is 3000 frames (240 s at 12.5 Hz).
# https://github.com/kyutai-labs/moshi/blob/main/moshi/moshi/models/loaders.py
# (`_lm_kwargs["context"] = 3000`)
# https://huggingface.co/docs/transformers/main/model_doc/moshi
# (`max_position_embeddings` / `sliding_window` default 3000)
MOSHI_MAX_HISTORY_FRAMES = 3000

# Callers that do not name a model get the PersonaPlex cap. Moshi deploys pass
# MOSHI_MAX_HISTORY_FRAMES via max_history_frames_for_repo.
MAX_HISTORY_FRAMES = PERSONAPLEX_MAX_HISTORY_FRAMES


def max_history_frames_for_repo(hf_repo: str) -> int:
    if "personaplex" in hf_repo:
        return PERSONAPLEX_MAX_HISTORY_FRAMES
    return MOSHI_MAX_HISTORY_FRAMES


@dataclass(frozen=True)
class PrefillDiagnostics:
    frame_count: int
    history_duration_sec: float
    elapsed_sec: float
    realtime_factor: float
    truncated_frames: int = 0


def iter_frames(pcm: np.ndarray, frame_samples: int) -> list[np.ndarray]:
    if pcm.shape[0] % frame_samples != 0:
        raise ValueError(
            f"pcm length {pcm.shape[0]} is not divisible by frame_samples {frame_samples}"
        )
    return [
        pcm[index : index + frame_samples]
        for index in range(0, pcm.shape[0], frame_samples)
    ]


def encode_frame(mimi, pcm_frame: np.ndarray, device: torch.device) -> torch.Tensor:
    chunk = torch.from_numpy(pcm_frame.astype(np.float32, copy=False))
    chunk = chunk.to(device=device)[None, None]
    return mimi.encode(chunk)


def decode_tokens(mimi, other_mimi, tokens: torch.Tensor) -> None:
    mimi.decode(tokens[:, 1:9])
    other_mimi.decode(tokens[:, 1:9])


async def run_duplex_prefill(
    user_mimi,
    agent_mimi,
    lm_gen,
    text_tokenizer,
    user_pcm: np.ndarray,
    assistant_pcm: np.ndarray,
    words: Sequence[Mapping[str, object]],
    device: torch.device,
    frame_samples: int,
    max_history_frames: int | None = None,
) -> PrefillDiagnostics:
    if user_pcm.shape[0] != assistant_pcm.shape[0]:
        raise ValueError("user_pcm and assistant_pcm must have equal length")

    user_frames = iter_frames(user_pcm, frame_samples)
    assistant_frames = iter_frames(assistant_pcm, frame_samples)
    history_words = parse_timed_words(words)
    text_tokens = aligned_text_tensor(
        history_words,
        frame_samples,
        user_pcm.shape[0],
        text_tokenizer,
        device,
    )
    # None reads the module global so tests can monkeypatch MAX_HISTORY_FRAMES.
    history_cap = (
        MAX_HISTORY_FRAMES if max_history_frames is None else max_history_frames
    )
    truncated_frames = max(0, len(user_frames) - history_cap)
    if truncated_frames:
        logger.warning("dropping %d history frames past context", truncated_frames)
        user_frames = user_frames[truncated_frames:]
        assistant_frames = assistant_frames[truncated_frames:]
        text_tokens = text_tokens[truncated_frames:]

    started = time.monotonic()
    for frame_idx, (user_frame, assistant_frame) in enumerate(
        zip(user_frames, assistant_frames, strict=True)
    ):
        # One frame is a full LM step plus four mimi passes, around realtime on
        # an A10G. Without yielding, a long history blocks the event loop past
        # the client's ping timeout and the connection dies mid-prefill.
        await asyncio.sleep(0)
        user_codes = encode_frame(user_mimi, user_frame, device)
        agent_codes = encode_frame(agent_mimi, assistant_frame, device)
        text_token = text_tokens[frame_idx].unsqueeze(0)
        tokens = lm_gen.step(user_codes, agent_codes, text_token)
        if tokens is None:
            continue
        decode_tokens(user_mimi, agent_mimi, tokens)

    elapsed = time.monotonic() - started
    sample_rate = getattr(lm_gen, "sample_rate", getattr(lm_gen, "_sample_rate", 24000))
    frame_count = len(user_frames)
    history_duration = frame_count * frame_samples / sample_rate
    realtime_factor = history_duration / elapsed if elapsed > 0 else float("inf")
    return PrefillDiagnostics(
        frame_count=frame_count,
        history_duration_sec=history_duration,
        elapsed_sec=elapsed,
        realtime_factor=realtime_factor,
        truncated_frames=truncated_frames,
    )
