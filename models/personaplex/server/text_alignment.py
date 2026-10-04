"""Map timed assistant words to the PersonaPlex text frame stream."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

import torch

from bench.audio import TARGET_SAMPLE_RATE

# PersonaPlex text stream specials: ['EPAD', 'BOS', 'EOS', 'PAD'].
PAD_TOKEN = 3
EPAD_TOKEN = 0


def frame_rate_hz(frame_samples: int, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    return sample_rate / frame_samples


def frame_count(sample_count: int, frame_samples: int) -> int:
    if sample_count % frame_samples != 0:
        raise ValueError(
            f"sample_count {sample_count} is not divisible by frame_samples {frame_samples}"
        )
    return sample_count // frame_samples


def tokenize_words(
    words: Sequence[object],
    tokenizer,
    frames_per_sec: float,
) -> list[tuple[int, list[int]]]:
    """Start frame and sentencepiece ids for each usable word."""
    tokenized: list[tuple[int, list[int]]] = []
    for word in words:
        text = getattr(word, "text", None)
        start_sec = getattr(word, "start_sec", None)
        if text is None or start_sec is None:
            continue
        pieces = [int(piece) for piece in tokenizer.encode(str(text))]
        if pieces:
            tokenized.append((max(0, int(float(start_sec) * frames_per_sec)), pieces))
    return tokenized


def align_words_to_frames(
    words: Sequence[object],
    frame_samples: int,
    sample_count: int,
    tokenizer,
) -> list[int]:
    """Lay words onto the 12.5 Hz text stream the way PersonaPlex was trained.

    One sentencepiece token per frame, words queued rather than overwritten when
    they land in the same frame, and EPAD marking the last pad frame before each
    run of word tokens. Mirrors `build_token_stream` in kyutai's moshi-finetune
    interleaver.
    """
    total_frames = frame_count(sample_count, frame_samples)
    aligned = [PAD_TOKEN] * total_frames
    tokenized = tokenize_words(words, tokenizer, frame_rate_hz(frame_samples))
    pending: deque[int] = deque()
    next_word = 0
    for frame_idx in range(total_frames):
        while next_word < len(tokenized) and tokenized[next_word][0] <= frame_idx:
            pending.extend(tokenized[next_word][1])
            next_word += 1
        if not pending:
            continue
        if frame_idx > 0 and aligned[frame_idx - 1] == PAD_TOKEN:
            aligned[frame_idx - 1] = EPAD_TOKEN
        aligned[frame_idx] = pending.popleft()
    return aligned


def aligned_text_tensor(
    words: Sequence[object],
    frame_samples: int,
    sample_count: int,
    tokenizer,
    device,
):
    aligned = align_words_to_frames(words, frame_samples, sample_count, tokenizer)
    return torch.tensor(aligned, dtype=torch.long, device=device)
