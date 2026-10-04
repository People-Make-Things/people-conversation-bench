"""Regression tests for the PCM frame accumulator.

The PersonaPlex server used to re-read `opus_reader.read_pcm()` into a fresh
local every poll, so anything shorter than one 80 ms model frame was dropped.
`drop_remainder_frames` reproduces that behaviour so these tests can show they
actually discriminate between the broken and fixed versions.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.audio import MODEL_FRAME_SAMPLES, FrameAccumulator

# Opus decodes in 20 ms units, so real reads are multiples of 480; the odd sizes
# cover future codec or transport changes.
READ_PATTERNS = {
    "one_opus_packet": [480] * 40,
    "two_opus_packets": [960] * 20,
    "aligned_model_frames": [1920] * 10,
    "one_and_a_half_frames": [2880] * 8,
    "single_samples": [1] * 6000,
    "prime_sized": [487, 1013, 1, 3169, 7, 4801, 89] * 3,
    "bursty_then_starved": [1920 * 5] + [1] * 400 + [1920 * 3],
    "stalled_reads": [0] * 50 + [480] * 8 + [0] * 50 + [1920 * 4],
}


def split_reads(pcm: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    reads: list[np.ndarray] = []
    offset = 0
    for size in sizes:
        reads.append(pcm[offset : offset + size])
        offset += size
        if offset >= pcm.shape[0]:
            break
    if offset < pcm.shape[0]:
        reads.append(pcm[offset:])
    return reads


def drop_remainder_frames(reads: list[np.ndarray], frame_samples: int) -> list[np.ndarray]:
    """Pre-fix behaviour: whatever is left under one frame never comes back."""
    frames: list[np.ndarray] = []
    for pcm in reads:
        while pcm.shape[-1] >= frame_samples:
            frames.append(pcm[:frame_samples])
            pcm = pcm[frame_samples:]
    return frames


def accumulate_frames(
    reads: list[np.ndarray],
    frame_samples: int,
) -> tuple[list[np.ndarray], FrameAccumulator]:
    accumulator = FrameAccumulator(frame_samples)
    frames: list[np.ndarray] = []
    for pcm in reads:
        frames.extend(accumulator.push(pcm))
    return frames, accumulator


def ramp(sample_count: int) -> np.ndarray:
    """Every sample distinct, so reordering is as detectable as loss."""
    return np.arange(sample_count, dtype=np.float32) / sample_count


@pytest.mark.parametrize("pattern", sorted(READ_PATTERNS))
def test_accumulator_conserves_every_sample_in_order(pattern: str) -> None:
    pcm = ramp(19200)
    frames, accumulator = accumulate_frames(
        split_reads(pcm, READ_PATTERNS[pattern]),
        MODEL_FRAME_SAMPLES,
    )
    assert all(frame.shape[0] == MODEL_FRAME_SAMPLES for frame in frames)
    assert accumulator.pending_samples < MODEL_FRAME_SAMPLES
    replayed = np.concatenate(frames + [accumulator.pending])
    assert replayed.tobytes() == pcm.tobytes()


@pytest.mark.parametrize("pattern", sorted(READ_PATTERNS))
def test_accumulator_beats_pre_fix_loop(pattern: str) -> None:
    pcm = ramp(19200)
    reads = split_reads(pcm, READ_PATTERNS[pattern])
    frames, accumulator = accumulate_frames(reads, MODEL_FRAME_SAMPLES)
    dropped = drop_remainder_frames(reads, MODEL_FRAME_SAMPLES)

    kept = len(frames) * MODEL_FRAME_SAMPLES + accumulator.pending_samples
    assert kept == pcm.shape[0]
    misaligned = any(read.shape[0] % MODEL_FRAME_SAMPLES for read in reads)
    if misaligned:
        assert len(dropped) * MODEL_FRAME_SAMPLES < pcm.shape[0]
    else:
        assert len(dropped) == len(frames)


def test_pre_fix_loop_discards_everything_below_one_frame() -> None:
    pcm = ramp(19200)
    reads = split_reads(pcm, [480] * 40)
    assert drop_remainder_frames(reads, MODEL_FRAME_SAMPLES) == []
    frames, accumulator = accumulate_frames(reads, MODEL_FRAME_SAMPLES)
    assert len(frames) == 10
    assert accumulator.pending_samples == 0


def test_accumulator_defaults_to_the_model_frame() -> None:
    accumulator = FrameAccumulator()
    assert accumulator.frame_samples == MODEL_FRAME_SAMPLES
    assert accumulator.push(np.zeros(MODEL_FRAME_SAMPLES - 1, dtype=np.float32)) == []
    assert len(accumulator.push(np.zeros(1, dtype=np.float32))) == 1


def test_accumulator_casts_reads_to_float32() -> None:
    accumulator = FrameAccumulator(4)
    frames = accumulator.push(np.arange(8, dtype=np.float64))
    assert [frame.dtype for frame in frames] == [np.dtype(np.float32)] * 2
