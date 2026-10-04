"""EOT response-length scoring for rollout traces."""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from bench.audio import write_wav_pcm16
from bench.protocol import TimedWord
from bench.speech import (
    SPEECH_MIN_DURATION_SEC,
    SPEECH_RMS_THRESHOLD,
    InputTimingEvent,
    observed_input_time,
    post_checkpoint_intervals,
)
from bench.trace import SCORED, RolloutTrace
from bench.transcribe import transcribe_words


@dataclass(frozen=True)
class ResponseLengthScore:
    eval: ClassVar[str] = "response-length"
    status: str
    human_duration_sec: float
    model_duration_sec: float
    duration_delta_sec: float
    duration_ratio: float | None
    human_word_count: int
    model_word_count: int
    word_delta: int
    word_ratio: float | None
    first_speech_sec: float | None
    last_speech_sec: float | None
    empty_response: bool
    timed_out: bool
    human_text: str = ""
    latency_ms: float | None = None
    wordless_phonation: bool = False

    def to_dict(self) -> dict:
        return {
            "eval": self.eval,
            "status": self.status,
            "human_text": self.human_text,
            "human_duration_sec": self.human_duration_sec,
            "model_duration_sec": self.model_duration_sec,
            "duration_delta_sec": self.duration_delta_sec,
            "duration_ratio": self.duration_ratio,
            "human_word_count": self.human_word_count,
            "model_word_count": self.model_word_count,
            "word_delta": self.word_delta,
            "word_ratio": self.word_ratio,
            "first_speech_sec": self.first_speech_sec,
            "last_speech_sec": self.last_speech_sec,
            "empty_response": self.empty_response,
            "timed_out": self.timed_out,
            "latency_ms": self.latency_ms,
            "wordless_phonation": self.wordless_phonation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResponseLengthScore:
        return cls(
            status=str(data.get("status", SCORED)),
            human_duration_sec=float(data["human_duration_sec"]),
            model_duration_sec=float(data["model_duration_sec"]),
            duration_delta_sec=float(data["duration_delta_sec"]),
            duration_ratio=data.get("duration_ratio"),
            human_word_count=int(data["human_word_count"]),
            model_word_count=int(data["model_word_count"]),
            word_delta=int(data["word_delta"]),
            word_ratio=data.get("word_ratio"),
            first_speech_sec=data.get("first_speech_sec"),
            last_speech_sec=data.get("last_speech_sec"),
            empty_response=bool(data["empty_response"]),
            timed_out=bool(data["timed_out"]),
            human_text=str(data.get("human_text", "")),
            latency_ms=(
                None if data.get("latency_ms") is None else float(data["latency_ms"])
            ),
            wordless_phonation=bool(data.get("wordless_phonation", False)),
        )


def compare_ratio(model_value: float, human_value: float) -> float | None:
    if human_value == 0:
        return None
    return model_value / human_value


def post_checkpoint_chunks(trace: RolloutTrace) -> list[tuple[float, np.ndarray]]:
    """Post-checkpoint playback slices: (playback start, pcm).

    A chunk that straddles the checkpoint is trimmed; gaps between chunks
    stay on the playback clock and are not filled.
    """
    checkpoint_sec = observed_input_time(trace.events, trace.checkpoint_sec)
    chunks: list[tuple[float, np.ndarray]] = []
    for audio, pcm in zip(trace.played_audio, trace.played_pcm, strict=True):
        if audio.end_sec <= checkpoint_sec:
            continue
        start_sec = audio.start_sec
        if audio.start_sec < checkpoint_sec:
            skip = int(round((checkpoint_sec - audio.start_sec) * trace.sample_rate))
            pcm = pcm[skip:]
            start_sec += skip / trace.sample_rate
        if pcm.size:
            chunks.append((start_sec, pcm))
    return chunks


def post_checkpoint_pcm(trace: RolloutTrace) -> np.ndarray:
    chunks = [pcm for _start, pcm in post_checkpoint_chunks(trace)]
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def model_response_words(trace: RolloutTrace) -> list[TimedWord]:
    pcm = post_checkpoint_pcm(trace)
    if pcm.size == 0:
        return []
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "model_response.wav"
        write_wav_pcm16(path, pcm, trace.sample_rate)
        return transcribe_words(path)


def first_word_playback_sec(
    trace: RolloutTrace, words: list[TimedWord]
) -> float | None:
    """Map the first word's clip time back onto the playback clock.

    Word timestamps live on the concatenated post-checkpoint clip that
    model_response_words transcribed; playback gaps between chunks were
    squeezed out of it. Walk the same slices to put the clock back.
    """
    if not words:
        return None
    offset = words[0].start_sec
    elapsed = 0.0
    for start_sec, pcm in post_checkpoint_chunks(trace):
        duration = pcm.size / trace.sample_rate
        if offset < elapsed + duration:
            return start_sec + (offset - elapsed)
        elapsed += duration
    # Whisper padded its last timestamp past the audio it was given.
    return None


def eot_onset(
    first_word_sec: float | None,
    events: Sequence[InputTimingEvent],
    checkpoint_sec: float,
    intervals: list[tuple[float, float]],
) -> tuple[float | None, bool]:
    """Wait from EOT to the onset of the first transcribed word.

    The onset is the first Whisper word's playback time — the same evidence
    every word metric scores on. The gate intervals cannot carry the onset
    themselves: a model that murmurs into its first word puts the murmur and
    the word in one interval, and the protocol text stream only says the model
    produced text somewhere, not when. Above-gate audio that transcribes to
    nothing is murmur, not a response. Returns (latency_ms,
    wordless_phonation).
    """
    if not intervals:
        return None, False
    if first_word_sec is None:
        return None, True
    observed = observed_input_time(events, checkpoint_sec)
    return (first_word_sec - observed) * 1000, False


def score_response_length(
    trace: RolloutTrace,
    *,
    human_duration_sec: float,
    human_word_count: int,
    model_word_count: int,
    first_word_sec: float | None,
    human_text: str = "",
    rms_threshold: float = SPEECH_RMS_THRESHOLD,
    min_duration_sec: float = SPEECH_MIN_DURATION_SEC,
) -> ResponseLengthScore:
    if trace.expected_action != "eot":
        raise ValueError(
            f"response-length scoring requires eot, got {trace.expected_action}"
        )
    intervals = post_checkpoint_intervals(
        trace.played_audio,
        trace.events,
        trace.checkpoint_sec,
        rms_threshold=rms_threshold,
        min_duration_sec=min_duration_sec,
    )
    first_speech_sec = intervals[0][0] if intervals else None
    last_speech_sec = intervals[-1][1] if intervals else None
    model_duration_sec = (
        last_speech_sec - first_speech_sec
        if first_speech_sec is not None and last_speech_sec is not None
        else 0.0
    )
    latency_ms, wordless_phonation = eot_onset(
        first_word_sec, trace.events, trace.checkpoint_sec, intervals
    )
    return ResponseLengthScore(
        status=trace.status,
        human_duration_sec=human_duration_sec,
        model_duration_sec=model_duration_sec,
        duration_delta_sec=model_duration_sec - human_duration_sec,
        duration_ratio=compare_ratio(model_duration_sec, human_duration_sec),
        human_word_count=human_word_count,
        model_word_count=model_word_count,
        word_delta=model_word_count - human_word_count,
        word_ratio=compare_ratio(float(model_word_count), float(human_word_count)),
        first_speech_sec=first_speech_sec,
        last_speech_sec=last_speech_sec,
        empty_response=not intervals,
        timed_out=trace.timed_out,
        human_text=human_text,
        latency_ms=latency_ms,
        wordless_phonation=wordless_phonation,
    )
