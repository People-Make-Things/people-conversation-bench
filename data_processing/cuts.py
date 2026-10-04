"""Turn-aligned conversation cuts for session chunking and recall prefixes.

Callers pass the preferred pack length and the hard window. Recall targets stay
an independent variable.
"""

from __future__ import annotations

from bench.audio import PERSONAPLEX_CONTEXT_SEC
from data_processing.labels import iter_turns

SPEAKERS = ("speaker_1", "speaker_2")

MAX_CHUNK_SEC = PERSONAPLEX_CONTEXT_SEC
MIN_CHUNK_SEC = 30.0
CLEAN_GAP_BUFFER_SEC = 0.3
RECALL_SLACK_SEC = 8.0
HISTORY_TARGETS_SEC = (30.0, 60.0, 90.0, 120.0)

EPS = 1e-6


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + EPS:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def conversation_end_sec(labels: dict, duration_sec: float | None = None) -> float:
    times: list[float] = []
    if duration_sec is not None:
        times.append(float(duration_sec))
    for speaker in SPEAKERS:
        for event in labels.get(speaker) or []:
            times.append(float(event["timestamp"]))
        for turn in iter_turns(labels, speaker):
            times.append(turn.end)
    for start, end in extra_busy_intervals(labels):
        times.append(end)
        times.append(start)
    return max(times) if times else 0.0


def extra_busy_intervals(labels: dict) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for item in labels.get("overlaps") or []:
        intervals.append((float(item["start"]), float(item["end"])))
    for item in labels.get("backchannels") or []:
        intervals.append((float(item["start"]), float(item["end"])))
    return intervals


def turn_intervals(labels: dict) -> list[tuple[float, float]]:
    return [
        (turn.start, turn.end)
        for speaker in SPEAKERS
        for turn in iter_turns(labels, speaker)
    ]


def busy_intervals(labels: dict) -> list[tuple[float, float]]:
    return merge_intervals(turn_intervals(labels) + extra_busy_intervals(labels))


def covers(time_sec: float, start: float, stop: float) -> bool:
    return start + EPS < time_sec < stop - EPS


def idle_intervals(
    labels: dict,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[tuple[float, float]]:
    end = conversation_end_sec(labels) if end_sec is None else end_sec
    if end <= start_sec + EPS:
        return []
    idle: list[tuple[float, float]] = []
    cursor = start_sec
    for busy_start, busy_end in busy_intervals(labels):
        if busy_end <= start_sec + EPS:
            continue
        if busy_start >= end - EPS:
            break
        gap_end = min(busy_start, end)
        if gap_end > cursor + EPS:
            idle.append((cursor, gap_end))
        cursor = max(cursor, min(busy_end, end))
        if cursor >= end - EPS:
            break
    if end > cursor + EPS:
        idle.append((cursor, end))
    return idle


def turn_ends(labels: dict) -> list[float]:
    ends = [turn.end for speaker in SPEAKERS for turn in iter_turns(labels, speaker)]
    return sorted(set(ends))


def is_clean_cut(labels: dict, time_sec: float, duration_sec: float | None = None) -> bool:
    if time_sec < -EPS:
        return False
    end = conversation_end_sec(labels, duration_sec)
    if time_sec > end + EPS:
        return False
    for start, stop in turn_intervals(labels) + extra_busy_intervals(labels):
        if covers(time_sec, start, stop):
            return False
    if any(abs(time_sec - end_sec) <= EPS for end_sec in turn_ends(labels)):
        return True
    for idle_start, idle_end in idle_intervals(labels, end_sec=end):
        if idle_end - idle_start + EPS < CLEAN_GAP_BUFFER_SEC:
            continue
        if idle_start - EPS <= time_sec <= idle_end + EPS:
            return True
    return False


def clean_cut_candidates(
    labels: dict,
    duration_sec: float | None = None,
) -> list[float]:
    end = conversation_end_sec(labels, duration_sec)
    times = set(turn_ends(labels))
    times.add(end)
    for idle_start, idle_end in idle_intervals(labels, end_sec=end):
        if idle_end - idle_start + EPS >= CLEAN_GAP_BUFFER_SEC:
            times.add(idle_start)
            times.add(idle_end)
    return sorted(time for time in times if is_clean_cut(labels, time, duration_sec))


def last_clean_cut(
    labels: dict,
    target_sec: float,
    slack_sec: float = RECALL_SLACK_SEC,
    duration_sec: float | None = None,
) -> float | None:
    lo = target_sec - slack_sec
    if is_clean_cut(labels, target_sec, duration_sec):
        return target_sec
    best: float | None = None
    for time_sec in clean_cut_candidates(labels, duration_sec):
        if lo - EPS <= time_sec <= target_sec + EPS:
            best = time_sec
    return best


def recall_history_cuts(
    labels: dict,
    duration_sec: float | None = None,
) -> list[tuple[float, float]]:
    cuts: list[tuple[float, float]] = []
    for target_sec in HISTORY_TARGETS_SEC:
        end_sec = last_clean_cut(
            labels,
            target_sec,
            slack_sec=RECALL_SLACK_SEC,
            duration_sec=duration_sec,
        )
        if end_sec is None:
            continue
        cuts.append((target_sec, end_sec))
    return cuts


def window_has_speech(labels: dict, start_sec: float, end_sec: float) -> bool:
    for speaker in SPEAKERS:
        for turn in iter_turns(labels, speaker):
            if turn.end > start_sec + EPS and turn.start < end_sec - EPS:
                return True
    return False


def next_cut(
    labels: dict,
    start_sec: float,
    end_sec: float,
    budget_sec: float,
    max_chunk_sec: float,
    min_duration_sec: float,
    duration_sec: float | None = None,
) -> float:
    hard = min(start_sec + max_chunk_sec, end_sec)
    preferred = min(start_sec + budget_sec, hard)
    cut = last_clean_cut(
        labels,
        preferred,
        slack_sec=max(preferred - start_sec - min_duration_sec, 0.0),
        duration_sec=duration_sec,
    )
    if cut is None or cut <= start_sec + EPS:
        cut = last_clean_cut(
            labels,
            hard,
            slack_sec=hard - start_sec,
            duration_sec=duration_sec,
        )
    if cut is None or cut <= start_sec + EPS:
        return hard
    return cut


def session_chunks(
    labels: dict,
    duration_sec: float | None = None,
    budget_sec: float | None = None,
    min_duration_sec: float = MIN_CHUNK_SEC,
    max_chunk_sec: float = MAX_CHUNK_SEC,
) -> list[tuple[float, float]]:
    end = conversation_end_sec(labels, duration_sec)
    if end <= EPS:
        return []
    pack_sec = max_chunk_sec if budget_sec is None else budget_sec
    chunks: list[tuple[float, float]] = []
    start = 0.0
    while start < end - EPS:
        cut = next_cut(
            labels,
            start,
            end,
            pack_sec,
            max_chunk_sec,
            min_duration_sec,
            duration_sec,
        )
        if cut <= start + EPS:
            break
        if window_has_speech(labels, start, cut):
            chunks.append((start, cut))
        start = cut
    return chunks
