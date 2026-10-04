"""Shared audio constants and PCM16 codec helpers for runtime and eval prep."""

from __future__ import annotations

import base64
import wave
from pathlib import Path

import numpy as np

TARGET_SAMPLE_RATE = 24000
# PersonaPlex mimi frame: 80 ms at 24 kHz.
MODEL_FRAME_SAMPLES = 1920
# Official trained window. Must match PERSONAPLEX_MAX_HISTORY_FRAMES in
# models/personaplex/server/prefill.py (that module cannot import bench).
PERSONAPLEX_MAX_HISTORY_FRAMES = 2048
PERSONAPLEX_CONTEXT_SEC = PERSONAPLEX_MAX_HISTORY_FRAMES * (
    MODEL_FRAME_SAMPLES / TARGET_SAMPLE_RATE
)


class FrameAccumulator:
    """Carry sub-frame PCM remainders across reads of arbitrary length.

    Opus decodes in 20 ms units while the model consumes 80 ms frames, so a
    read that is not a whole number of frames must keep its tail for the next
    read instead of discarding it.
    """

    def __init__(self, frame_samples: int = MODEL_FRAME_SAMPLES):
        self.frame_samples = frame_samples
        self.pending = np.zeros(0, dtype=np.float32)

    def push(self, pcm: np.ndarray) -> list[np.ndarray]:
        """Buffer pcm and return every whole frame that is now complete."""
        self.pending = np.concatenate(
            [self.pending, pcm.astype(np.float32, copy=False)]
        )
        complete = (self.pending.shape[0] // self.frame_samples) * self.frame_samples
        frames = [
            self.pending[index : index + self.frame_samples]
            for index in range(0, complete, self.frame_samples)
        ]
        self.pending = self.pending[complete:]
        return frames

    def flush(self) -> np.ndarray:
        """Zero-pad and return the remainder, so a closing stream loses nothing."""
        remainder = self.pending
        self.pending = np.zeros(0, dtype=np.float32)
        if remainder.shape[0] == 0:
            return remainder
        return pad_pcm_to_model_frames(remainder, self.frame_samples)

    @property
    def pending_samples(self) -> int:
        return self.pending.shape[0]


def pcm_float_to_int16_bytes(pcm: np.ndarray) -> bytes:
    # Round rather than truncate, and scale by the same 32768 the decode side
    # divides by, so a round trip is symmetric to within half a step.
    scaled = np.rint(np.clip(pcm, -1.0, 1.0) * 32768.0)
    return np.clip(scaled, -32768, 32767).astype(np.int16, copy=False).tobytes()


def pcm16_bytes_to_float(raw: bytes) -> np.ndarray:
    if len(raw) % 2 != 0:
        raise ValueError("PCM16 payload must contain an even number of bytes")
    pcm16 = np.frombuffer(raw, dtype=np.int16)
    return (pcm16.astype(np.float32) / 32768.0).astype(np.float32, copy=False)


def pcm_float_to_b64(pcm: np.ndarray) -> str:
    return base64.b64encode(pcm_float_to_int16_bytes(pcm)).decode("ascii")


def b64_to_pcm_float(encoded: str) -> np.ndarray:
    return pcm16_bytes_to_float(base64.b64decode(encoded))


def append_pcm16_buffer(existing: np.ndarray | None, payload: bytes) -> np.ndarray:
    chunk = pcm16_bytes_to_float(payload)
    if existing is None:
        return chunk
    return np.concatenate([existing, chunk]).astype(np.float32, copy=False)


def resample(pcm: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return pcm
    ratio = from_rate / to_rate
    if ratio.is_integer():
        return pcm[:: int(ratio)].astype(np.float32, copy=False)

    target_length = int(round(pcm.shape[0] * to_rate / from_rate))
    if target_length <= 0:
        return np.zeros(0, dtype=np.float32)
    source_positions = np.arange(target_length, dtype=np.float64) * from_rate / to_rate
    floor = np.floor(source_positions).astype(np.int64)
    ceil = np.minimum(floor + 1, pcm.shape[0] - 1)
    weight = (source_positions - floor).astype(np.float32)
    return (
        pcm[floor] * (1.0 - weight) + pcm[ceil] * weight
    ).astype(np.float32, copy=False)


def read_wav_segment(
    path: Path,
    end_sec: float | None = None,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        if sample_width != 2:
            raise ValueError(f"{path}: expected 16-bit PCM, got {sample_width * 8}-bit")

        end_frame = handle.getnframes()
        if end_sec is not None:
            end_frame = min(end_frame, int(end_sec * sample_rate))
        if end_frame <= 0:
            return np.zeros(0, dtype=np.float32), target_rate

        raw = handle.readframes(end_frame)

    pcm = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.int16, copy=False)

    pcm_float = (pcm.astype(np.float32) / 32768.0).astype(np.float32, copy=False)
    pcm_float = resample(pcm_float, sample_rate, target_rate)
    return pcm_float, target_rate


def pad_pcm_to_model_frames(
    pcm: np.ndarray,
    frame_samples: int = MODEL_FRAME_SAMPLES,
) -> np.ndarray:
    remainder = pcm.shape[0] % frame_samples
    if remainder == 0:
        return pcm
    return np.pad(pcm, (0, frame_samples - remainder)).astype(np.float32, copy=False)


def write_wav_pcm16(path: Path, pcm: np.ndarray, sample_rate: int) -> None:
    if pcm.ndim == 1:
        channels = 1
    elif pcm.ndim == 2:
        channels = pcm.shape[1]
    else:
        raise ValueError("WAV audio must be a 1-D or 2-D array")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_float_to_int16_bytes(pcm))
