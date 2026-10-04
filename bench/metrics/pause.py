"""Pause-recognition scoring for rollout traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from bench.speech import (
    SPEECH_MIN_DURATION_SEC,
    SPEECH_RELEASE_RATIO,
    SPEECH_RMS_THRESHOLD,
    observed_input_time,
    speech_intervals,
)
from bench.trace import NO_MODEL_AUDIO, SCORED, RolloutTrace

TAKEOVER = "Takeover"
HOLD = "Hold"
WORDLESS = "Wordless"
NO_AUDIO = "No audio"

# Triage flag for above-gate audio while the model's text stream stayed empty.
MAX_WORDLESS_SPEECH_RATIO = 0.05


@dataclass(frozen=True)
class PauseRecognitionScore:
    eval: ClassVar[str] = "pause-recognition"
    status: str
    score: int | None
    false_takeover: bool
    takeover_latency_ms: float | None
    first_speech_sec: float | None
    labeled_pause_duration_ms: float | None
    model_wait_from_pause_start_ms: float | None
    model_wait_minus_labeled_pause_ms: float | None
    rms_threshold: float
    release_ratio: float
    min_duration_sec: float
    model_audio_sec: float
    speech_continued_from_before_checkpoint: bool
    speech_ratio: float
    wordless_phonation: bool

    def to_dict(self) -> dict:
        return {
            "eval": self.eval,
            "status": self.status,
            "score": self.score,
            "false_takeover": self.false_takeover,
            "takeover_latency_ms": self.takeover_latency_ms,
            "first_speech_sec": self.first_speech_sec,
            "labeled_pause_duration_ms": self.labeled_pause_duration_ms,
            "model_wait_from_pause_start_ms": self.model_wait_from_pause_start_ms,
            "model_wait_minus_labeled_pause_ms": self.model_wait_minus_labeled_pause_ms,
            "rms_threshold": self.rms_threshold,
            "release_ratio": self.release_ratio,
            "min_duration_sec": self.min_duration_sec,
            "model_audio_sec": self.model_audio_sec,
            "speech_continued_from_before_checkpoint": (
                self.speech_continued_from_before_checkpoint
            ),
            "speech_ratio": self.speech_ratio,
            "wordless_phonation": self.wordless_phonation,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PauseRecognitionScore:
        score = data.get("score")
        return cls(
            status=str(data.get("status", SCORED)),
            score=None if score is None else int(score),
            false_takeover=bool(data["false_takeover"]),
            takeover_latency_ms=data.get("takeover_latency_ms"),
            first_speech_sec=data.get("first_speech_sec"),
            labeled_pause_duration_ms=data.get("labeled_pause_duration_ms"),
            model_wait_from_pause_start_ms=data.get("model_wait_from_pause_start_ms"),
            model_wait_minus_labeled_pause_ms=data.get(
                "model_wait_minus_labeled_pause_ms"
            ),
            rms_threshold=float(data.get("rms_threshold", SPEECH_RMS_THRESHOLD)),
            release_ratio=float(data.get("release_ratio", SPEECH_RELEASE_RATIO)),
            min_duration_sec=float(
                data.get("min_duration_sec", SPEECH_MIN_DURATION_SEC)
            ),
            model_audio_sec=float(data.get("model_audio_sec", 0.0)),
            speech_continued_from_before_checkpoint=bool(
                data.get("speech_continued_from_before_checkpoint", False)
            ),
            speech_ratio=float(data.get("speech_ratio", 0.0)),
            wordless_phonation=bool(data.get("wordless_phonation", False)),
        )


def pause_outcome(row: PauseRecognitionScore) -> str:
    if row.status == NO_MODEL_AUDIO:
        return NO_AUDIO
    if row.false_takeover:
        return TAKEOVER
    if row.wordless_phonation:
        return WORDLESS
    return HOLD


def score_pause_recognition(
    trace: RolloutTrace,
    rms_threshold: float = SPEECH_RMS_THRESHOLD,
    min_duration_sec: float = SPEECH_MIN_DURATION_SEC,
    release_ratio: float = SPEECH_RELEASE_RATIO,
) -> PauseRecognitionScore:
    if trace.expected_action != "pause":
        raise ValueError(
            f"pause-recognition scoring requires pause, got {trace.expected_action}"
        )
    intervals = speech_intervals(
        trace.played_audio,
        rms_threshold=rms_threshold,
        min_duration_sec=min_duration_sec,
        release_ratio=release_ratio,
    )
    checkpoint_sec = observed_input_time(trace.events, trace.checkpoint_sec)
    window_end_sec = observed_input_time(trace.events, trace.window_end_sec)
    labeled_pause_duration_ms = (
        trace.window_end_sec - trace.checkpoint_sec
    ) * 1000
    # White-noise / murmur above the RMS gate is not a response. A takeover
    # and a measured wait both require the model to have uttered words.
    response_intervals = intervals if trace.text.strip() else []
    first_takeover_speech = next(
        (
            max(start, checkpoint_sec)
            for start, end in response_intervals
            if start < window_end_sec and end > checkpoint_sec
        ),
        None,
    )
    false_takeover = first_takeover_speech is not None
    takeover_latency = (
        (first_takeover_speech - checkpoint_sec) * 1000
        if first_takeover_speech is not None
        else None
    )
    first_response_speech = next(
        (start for start, _ in response_intervals if start >= checkpoint_sec),
        None,
    )
    model_wait_from_pause_start_ms = (
        (first_response_speech - checkpoint_sec) * 1000
        if first_response_speech is not None
        else None
    )
    model_wait_minus_labeled_pause_ms = (
        model_wait_from_pause_start_ms - labeled_pause_duration_ms
        if model_wait_from_pause_start_ms is not None
        else None
    )
    scored = trace.status == SCORED
    speech_ratio = (
        sum(end - start for start, end in intervals) / trace.model_audio_sec
        if trace.model_audio_sec > 0
        else 0.0
    )
    return PauseRecognitionScore(
        status=trace.status,
        score=(0 if false_takeover else 1) if scored else None,
        false_takeover=false_takeover,
        takeover_latency_ms=takeover_latency,
        first_speech_sec=first_takeover_speech,
        labeled_pause_duration_ms=labeled_pause_duration_ms,
        model_wait_from_pause_start_ms=model_wait_from_pause_start_ms,
        model_wait_minus_labeled_pause_ms=model_wait_minus_labeled_pause_ms,
        rms_threshold=rms_threshold,
        release_ratio=release_ratio,
        min_duration_sec=min_duration_sec,
        model_audio_sec=trace.model_audio_sec,
        speech_continued_from_before_checkpoint=any(
            start < checkpoint_sec < end for start, end in intervals
        ),
        speech_ratio=speech_ratio,
        wordless_phonation=(
            not trace.text.strip() and speech_ratio > MAX_WORDLESS_SPEECH_RATIO
        ),
    )
