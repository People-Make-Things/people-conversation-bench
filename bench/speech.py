"""RMS speech detection and source-time mapping for rollout traces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

# Shared by the drain and by pause/EOT scoring. Calibration is in docs/phonation.md.
SPEECH_RMS_THRESHOLD = 0.01
SPEECH_MIN_DURATION_SEC = 0.08
SPEECH_MAX_GAP_SEC = 0.04
SPEECH_RELEASE_RATIO = 0.5


@dataclass(frozen=True)
class PlayedAudio:
    start_sec: float
    end_sec: float
    rms: float


class InputTimingEvent(Protocol):
    kind: str
    at_sec: float
    data: dict


def observed_input_time(
    events: Sequence[InputTimingEvent],
    source_time_sec: float,
) -> float:
    frames = [event for event in events if event.kind == "input_audio_sent"]
    if not frames:
        return source_time_sec
    earlier = [
        event
        for event in frames
        if float(event.data.get("scheduled_at_sec", 0.0)) <= source_time_sec
    ]
    # Before the first frame there is still a source-to-wall-clock mapping to
    # extrapolate from; returning source time here would mix the two clocks.
    frame = earlier[-1] if earlier else frames[0]
    scheduled_at = float(frame.data["scheduled_at_sec"])
    return frame.at_sec + source_time_sec - scheduled_at


def speech_intervals(
    audio: list[PlayedAudio],
    rms_threshold: float = SPEECH_RMS_THRESHOLD,
    min_duration_sec: float = SPEECH_MIN_DURATION_SEC,
    max_gap_sec: float = SPEECH_MAX_GAP_SEC,
    release_ratio: float = SPEECH_RELEASE_RATIO,
) -> list[tuple[float, float]]:
    """Gate speech with hysteresis: open at rms_threshold, hold to a fraction of it.

    A single quiet block inside a word is a dip in the envelope, not the end of
    speech. Closing on it and then discarding the short fragments that leaves
    biases every pause toward a pass.
    """
    release_threshold = rms_threshold * release_ratio
    intervals: list[tuple[float, float]] = []
    start: float | None = None
    end = 0.0
    for segment in audio:
        threshold = release_threshold if start is not None else rms_threshold
        if segment.rms < threshold:
            continue
        if start is not None and segment.start_sec - end > max_gap_sec:
            intervals.append((start, end))
            start = None
        if start is None:
            start = segment.start_sec
        end = segment.end_sec
    if start is not None:
        intervals.append((start, end))
    return [
        (start, end) for start, end in intervals if end - start >= min_duration_sec
    ]


def last_speech_end_sec(
    audio: list[PlayedAudio],
    after_sec: float = 0.0,
    *,
    rms_threshold: float = SPEECH_RMS_THRESHOLD,
    min_duration_sec: float = SPEECH_MIN_DURATION_SEC,
    max_gap_sec: float = SPEECH_MAX_GAP_SEC,
) -> float | None:
    intervals = speech_intervals(
        audio,
        rms_threshold=rms_threshold,
        min_duration_sec=min_duration_sec,
        max_gap_sec=max_gap_sec,
    )
    ends = [end for _, end in intervals if end > after_sec]
    if not ends:
        return None
    return ends[-1]


def post_checkpoint_intervals(
    audio: list[PlayedAudio],
    events: Sequence[InputTimingEvent],
    checkpoint_sec: float,
    rms_threshold: float = SPEECH_RMS_THRESHOLD,
    min_duration_sec: float = SPEECH_MIN_DURATION_SEC,
) -> list[tuple[float, float]]:
    observed = observed_input_time(events, checkpoint_sec)
    post: list[tuple[float, float]] = []
    for start, end in speech_intervals(
        audio,
        rms_threshold=rms_threshold,
        min_duration_sec=min_duration_sec,
    ):
        if end > observed:
            post.append((max(start, observed), end))
    return post
