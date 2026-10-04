"""WAV slicing helpers built on shared bench audio utilities."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

from bench.audio import TARGET_SAMPLE_RATE, read_wav_segment, resample, write_wav_pcm16

WAVE_FORMAT_PCM = 1
WAVE_FORMAT_IEEE_FLOAT = 3


def read_wav_range(
    path: Path,
    start_sec: float,
    end_sec: float,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> tuple:
    pcm, sample_rate = read_wav_segment(path, end_sec, target_rate)
    if start_sec <= 0:
        return pcm, sample_rate
    start_sample = min(int(start_sec * sample_rate), pcm.shape[0])
    return pcm[start_sample:], sample_rate


def wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def write_conversation_mix(datapoint_dir: Path) -> Path:
    """Stereo listen mix: left speaker_1, right speaker_2."""
    left, sample_rate = read_wav_segment(datapoint_dir / "waves" / "speaker_1.wav")
    right, right_rate = read_wav_segment(datapoint_dir / "waves" / "speaker_2.wav")
    if right_rate != sample_rate:
        raise ValueError("speaker wavs must share a sample rate")
    length = max(left.shape[0], right.shape[0])
    stereo = np.zeros((length, 2), dtype=np.float32)
    stereo[: left.shape[0], 0] = left
    stereo[: right.shape[0], 1] = right
    dest = datapoint_dir / "conversation.wav"
    write_wav_pcm16(dest, stereo, sample_rate)
    return dest


def read_source_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read a mono WAV as float32. Accepts PCM16 or IEEE float32."""
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) < 12:
            raise ValueError(f"{path}: truncated WAV header")
        riff, _, wave_tag = struct.unpack("<4sI4s", header)
        if riff != b"RIFF" or wave_tag != b"WAVE":
            raise ValueError(f"{path}: not a RIFF/WAVE file")

        audio_format = None
        channels = None
        sample_rate = None
        bits = None
        data = None
        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                break
            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            payload = handle.read(chunk_size)
            if chunk_size % 2 == 1:
                handle.read(1)
            if chunk_id == b"fmt ":
                audio_format, channels, sample_rate, _, _, bits = struct.unpack(
                    "<HHIIHH", payload[:16]
                )
            elif chunk_id == b"data":
                data = payload

    if data is None or audio_format is None or channels is None:
        raise ValueError(f"{path}: missing fmt or data chunk")
    if channels < 1:
        raise ValueError(f"{path}: invalid channel count {channels}")

    if audio_format == WAVE_FORMAT_PCM and bits == 16:
        pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    elif audio_format == WAVE_FORMAT_IEEE_FLOAT and bits == 32:
        pcm = np.frombuffer(data, dtype="<f4").astype(np.float32, copy=False)
    else:
        raise ValueError(
            f"{path}: unsupported WAV format {audio_format}/{bits}-bit"
        )

    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1).astype(np.float32, copy=False)
    return pcm.astype(np.float32, copy=False), int(sample_rate)


def write_eval_wav(
    dest: Path,
    pcm: np.ndarray,
    sample_rate: int,
    target_rate: int = TARGET_SAMPLE_RATE,
) -> None:
    write_wav_pcm16(dest, resample(pcm, sample_rate, target_rate), target_rate)
