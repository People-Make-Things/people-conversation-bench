"""Parse turn-taking annotations into eval split events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SPLIT_TYPES = frozenset({"end_of_turn", "pause_start"})


@dataclass(frozen=True)
class SplitEvent:
    index: int
    timestamp: float
    split_type: str
    active_speaker: str
    model_speaker: str


@dataclass(frozen=True)
class Turn:
    speaker: str
    start: float
    end: float


@dataclass(frozen=True)
class RolloutWindow:
    turn_start: float
    input_end: float
    checkpoint: float
    window_end: float


def other_speaker(speaker: str) -> str:
    if speaker == "speaker_1":
        return "speaker_2"
    if speaker == "speaker_2":
        return "speaker_1"
    raise ValueError(f"unknown speaker: {speaker}")


def load_labels(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        labels = json.load(handle)
    if "merged" not in labels:
        raise ValueError(f"{path}: missing 'merged' timeline")
    return labels


def iter_turns(labels: dict, speaker: str) -> list[Turn]:
    events = labels.get(speaker)
    if not isinstance(events, list):
        raise ValueError(f"labels missing speaker timeline: {speaker}")

    turns: list[Turn] = []
    start: float | None = None
    for event in events:
        event_type = event.get("type")
        timestamp = float(event["timestamp"])
        if event_type == "voice_activation":
            start = timestamp
        elif event_type == "end_of_turn" and start is not None:
            turns.append(Turn(speaker=speaker, start=start, end=timestamp))
            start = None
    return turns


def current_turn_start(labels: dict, speaker: str, split_timestamp: float) -> float:
    start = 0.0
    for event in labels.get(speaker, []):
        timestamp = float(event["timestamp"])
        if timestamp > split_timestamp:
            break
        if event.get("type") == "voice_activation":
            start = timestamp
    return start


def next_speaker_event(
    labels: dict,
    speaker: str,
    event_type: str,
    after: float,
) -> float | None:
    for event in labels.get(speaker, []):
        timestamp = float(event["timestamp"])
        if timestamp > after and event.get("type") == event_type:
            return timestamp
    return None


def next_turn_after(labels: dict, speaker: str, after_sec: float) -> Turn | None:
    for turn in iter_turns(labels, speaker):
        if turn.start >= after_sec - 1e-6:
            return turn
    return None


def rollout_window(labels: dict, event: SplitEvent) -> RolloutWindow:
    turn_start = current_turn_start(labels, event.active_speaker, event.timestamp)
    if event.split_type == "end_of_turn":
        return RolloutWindow(
            turn_start=turn_start,
            input_end=event.timestamp,
            checkpoint=event.timestamp,
            window_end=event.timestamp,
        )

    pause_end = next_speaker_event(
        labels,
        event.active_speaker,
        "pause_end",
        event.timestamp,
    )
    if pause_end is None:
        raise ValueError(
            f"pause at {event.timestamp:.3f}s has no later pause_end "
            f"for {event.active_speaker}"
        )
    return RolloutWindow(
        turn_start=turn_start,
        input_end=pause_end,
        checkpoint=event.timestamp,
        window_end=pause_end,
    )


def iter_completed_turns(labels: dict, split_event: SplitEvent) -> list[Turn]:
    split_timestamp = split_event.timestamp
    completed: list[Turn] = []
    for speaker in ("speaker_1", "speaker_2"):
        for turn in iter_turns(labels, speaker):
            if turn.end < split_timestamp:
                completed.append(turn)
                continue
            if (
                turn.end == split_timestamp
                and split_event.split_type == "end_of_turn"
                and speaker == split_event.active_speaker
            ):
                continue
            if turn.end == split_timestamp:
                completed.append(turn)
    return sorted(completed, key=lambda turn: turn.start)


def history_start_sec(
    labels: dict,
    split_event: SplitEvent,
    turn_start: float,
    cap_sec: float,
) -> float:
    """Latest cut keeping history under cap_sec, snapped to a turn start.

    Snapping to a completed-turn boundary keeps the audio history and the text
    context describing the same turns instead of starting mid-turn. When no
    turn starts inside the budget (one very long turn), fall back to the raw
    cut so the cap still holds.
    """
    earliest = turn_start - cap_sec
    if earliest <= 0:
        return 0.0
    starts = [
        turn.start
        for turn in iter_completed_turns(labels, split_event)
        if earliest <= turn.start <= turn_start
    ]
    return min(starts, default=earliest)


def iter_prefix_turns(labels: dict, end_sec: float) -> list[Turn]:
    """Completed turns whose end is at or before end_sec, in start order."""
    completed: list[Turn] = []
    for speaker in ("speaker_1", "speaker_2"):
        for turn in iter_turns(labels, speaker):
            if turn.end <= end_sec + 1e-6:
                completed.append(turn)
    return sorted(completed, key=lambda turn: turn.start)


def prefix_end_sec(labels: dict, target_sec: float) -> float | None:
    """Last completed-turn end at or before target_sec, or None if none exist."""
    turns = iter_prefix_turns(labels, target_sec)
    if not turns:
        return None
    return max(turn.end for turn in turns)


def slice_interval(
    start_sec: float,
    end_sec: float,
    window_start: float,
    window_end: float,
) -> tuple[float, float] | None:
    lo = max(start_sec, window_start)
    hi = min(end_sec, window_end)
    if hi <= lo + 1e-6:
        return None
    return (lo - window_start, hi - window_start)


def rebase_timestamp_events(
    events: list[dict] | None,
    start_sec: float,
    end_sec: float,
) -> list[dict]:
    rebased = []
    for event in events or []:
        timestamp = float(event["timestamp"])
        if start_sec - 1e-6 <= timestamp <= end_sec + 1e-6:
            item = dict(event)
            item["timestamp"] = timestamp - start_sec
            rebased.append(item)
    return rebased


def slice_span_records(
    items: list[dict] | None,
    start_sec: float,
    end_sec: float,
) -> list[dict]:
    sliced = []
    for item in items or []:
        clipped = slice_interval(
            float(item["start"]),
            float(item["end"]),
            start_sec,
            end_sec,
        )
        if clipped is None:
            continue
        copied = dict(item)
        copied["start"] = clipped[0]
        copied["end"] = clipped[1]
        sliced.append(copied)
    return sliced


def slice_labels(labels: dict, start_sec: float, end_sec: float) -> dict:
    merged = []
    for index, event in enumerate(labels.get("merged") or []):
        timestamp = float(event["timestamp"])
        if start_sec - 1e-6 <= timestamp <= end_sec + 1e-6:
            item = dict(event)
            item["timestamp"] = timestamp - start_sec
            item["index"] = event.get("index", index)
            merged.append(item)
    return {
        "speaker_1": rebase_timestamp_events(labels.get("speaker_1"), start_sec, end_sec),
        "speaker_2": rebase_timestamp_events(labels.get("speaker_2"), start_sec, end_sec),
        "merged": merged,
        "overlaps": slice_span_records(labels.get("overlaps"), start_sec, end_sec),
        "backchannels": slice_span_records(
            labels.get("backchannels"), start_sec, end_sec
        ),
    }


def slice_words(
    channel_words: dict[str, list[dict]] | None,
    start_sec: float,
    end_sec: float,
) -> dict[str, list[dict]] | None:
    if not channel_words:
        return channel_words
    return {
        speaker: slice_span_records(words, start_sec, end_sec)
        for speaker, words in channel_words.items()
    }


def iter_split_events(labels: dict) -> list[SplitEvent]:
    events: list[SplitEvent] = []
    for index, event in enumerate(labels["merged"]):
        split_type = event.get("type")
        if split_type not in SPLIT_TYPES:
            continue
        active_speaker = event.get("id")
        if not active_speaker:
            raise ValueError(f"merged[{index}]: split event missing speaker id")
        events.append(
            SplitEvent(
                index=int(event["index"]) if "index" in event else index,
                timestamp=float(event["timestamp"]),
                split_type=split_type,
                active_speaker=active_speaker,
                model_speaker=other_speaker(active_speaker),
            )
        )
    return events
