"""Select eval-worthy split events with label heuristics and audio checks.

No model calls. Audio is measured through bench.audio_metrics, the same
hysteresis gate eval scoring uses, so an event kept here cannot later fail
scoring for audio reasons the gate would have caught at prepare time.

Every split event gets a Candidate with features, a keep/reject decision, and
a stable snake_case reason. Rejects are hard gates; survivors are ranked and
capped per source so one chatty session cannot flood the eval set.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import boto3

from bench.audio_metrics import speech_ratio
from data_processing.audio import read_wav_range, wav_duration_sec
from data_processing.curation_report import write_curation_figures
from data_processing.labels import (
    RolloutWindow,
    SplitEvent,
    iter_split_events,
    load_labels,
    next_turn_after,
    rollout_window,
)
from data_processing.sources import get_source, resolve_datapoint

# Hard gates: a candidate failing any of these is rejected outright.
MIN_PAUSE_SEC = 0.6
MIN_TURN_CONTEXT_SEC = 1.0
MIN_USER_TURN_SEC = 1.0
MIN_USER_TURN_WORDS = 3
MIN_REFERENCE_WORDS = 5
# A "reference reply" that starts long after the user turn is not a response
# to it; inspection found gaps up to 166 s where the reply continued an
# unrelated thread.
MAX_REFERENCE_GAP_SEC = 3.0
MAX_HOLD_SPEECH_RATIO = 0.25
MIN_TURN_SPEECH_RATIO = 0.2
MIN_REFERENCE_SPEECH_RATIO = 0.2

# Ranking: each feature is capped, normalized to [0, 1], then weighted.
PAUSE_SEC_CAP = 3.0
CONTEXT_SEC_CAP = 10.0
USER_TURN_SEC_CAP = 20.0
REFERENCE_WORDS_CAP = 40
REFERENCE_SEC_CAP = 15.0
GAP_SEC_CAP = 4.0
LEAD_WINDOW_SEC = 3.0
DEFAULT_KEEP_PER_TYPE = 3
# Sessions are downloaded and scored independently; the work is S3-bound.
CURATION_WORKERS = 6

# A pause right after one of these words is an obvious mid-clause hold. A
# pause after any other word could plausibly be an end of turn, which is the
# ambiguity that makes a pause point worth evaluating.
CONTINUATION_WORDS = frozenset(
    "and but or so because if then the a an to of in on at with for "
    "um uh er ah like that i my your this".split()
)


def thresholds() -> dict:
    return {
        "min_pause_sec": MIN_PAUSE_SEC,
        "min_turn_context_sec": MIN_TURN_CONTEXT_SEC,
        "min_user_turn_sec": MIN_USER_TURN_SEC,
        "min_user_turn_words": MIN_USER_TURN_WORDS,
        "min_reference_words": MIN_REFERENCE_WORDS,
        "max_reference_gap_sec": MAX_REFERENCE_GAP_SEC,
        "max_hold_speech_ratio": MAX_HOLD_SPEECH_RATIO,
        "min_turn_speech_ratio": MIN_TURN_SPEECH_RATIO,
        "min_reference_speech_ratio": MIN_REFERENCE_SPEECH_RATIO,
    }


@dataclass
class Candidate:
    event: SplitEvent
    features: dict
    keep: bool
    reason: str
    score: float | None = None

    def to_dict(self) -> dict:
        return {
            "index": self.event.index,
            "timestamp": self.event.timestamp,
            "split_type": self.event.split_type,
            "active_speaker": self.event.active_speaker,
            "features": self.features,
            "score": self.score,
            "keep": self.keep,
            "reason": self.reason,
        }


def split_exceeds_source_audio(
    event: SplitEvent,
    speaker_paths: dict[str, Path],
) -> str | None:
    for speaker in (event.active_speaker, event.model_speaker):
        duration_sec = wav_duration_sec(speaker_paths[speaker])
        if event.timestamp > duration_sec + 1e-6:
            return "split_beyond_source_audio"
    return None


def skip_reason(
    event: SplitEvent,
    labels: dict,
    speaker_paths: dict[str, Path],
) -> str | None:
    """Structural reasons an event cannot become an eval point at all."""
    # rollout_window raises when a pause_start has no later pause_end, which
    # real label files do contain; that is a reject, not a crash.
    try:
        window = rollout_window(labels, event)
    except ValueError:
        return "no_pause_end"
    audio_reason = split_exceeds_source_audio(event, speaker_paths)
    if audio_reason is not None:
        return audio_reason
    source_duration = wav_duration_sec(speaker_paths[event.active_speaker])
    if window.window_end > source_duration + 1e-6:
        return "pause_end_beyond_source_audio"
    if event.split_type != "end_of_turn":
        return None
    reference_turn = next_turn_after(labels, event.model_speaker, event.timestamp)
    if reference_turn is None or reference_turn.end <= reference_turn.start:
        return "no_reference_turn"
    assistant_duration = wav_duration_sec(speaker_paths[event.model_speaker])
    if reference_turn.end > assistant_duration + 1e-6:
        return "reference_beyond_source_audio"
    return None


def words_in_span(
    channel_words: dict[str, list[dict]] | None,
    speaker: str,
    start_sec: float,
    end_sec: float,
) -> list[dict]:
    if not channel_words:
        return []
    return [
        word
        for word in channel_words.get(speaker, [])
        if word["start"] < end_sec and word["end"] > start_sec
    ]


def channel_speech_ratio(path: Path, start_sec: float, end_sec: float) -> float:
    pcm, sample_rate = read_wav_range(path, start_sec, end_sec)
    return speech_ratio(pcm, sample_rate)


def evaluate_pause(
    event: SplitEvent,
    window: RolloutWindow,
    speaker_paths: dict[str, Path],
    channel_words: dict[str, list[dict]] | None,
) -> Candidate:
    pause_sec = window.window_end - window.checkpoint
    context_sec = window.checkpoint - window.turn_start
    last_word = None
    before = words_in_span(
        channel_words, event.active_speaker, window.turn_start, window.checkpoint
    )
    if before:
        last_word = before[-1]["text"].strip(".,!?").lower()
    features = {
        "pause_sec": round(pause_sec, 3),
        "context_sec": round(context_sec, 3),
        "last_word": last_word,
    }
    if pause_sec < MIN_PAUSE_SEC:
        return Candidate(event, features, False, "pause_too_short")
    if context_sec < MIN_TURN_CONTEXT_SEC:
        return Candidate(event, features, False, "short_turn_context")
    if words_in_span(
        channel_words, event.model_speaker, window.checkpoint, window.window_end
    ):
        return Candidate(event, features, False, "other_speaker_in_hold")

    active_path = speaker_paths[event.active_speaker]
    hold_ratio = channel_speech_ratio(active_path, window.checkpoint, window.window_end)
    features["hold_speech_ratio"] = round(hold_ratio, 3)
    if hold_ratio > MAX_HOLD_SPEECH_RATIO:
        return Candidate(event, features, False, "hold_gates_as_speech")
    lead_start = max(window.turn_start, window.checkpoint - LEAD_WINDOW_SEC)
    lead_ratio = channel_speech_ratio(active_path, lead_start, window.checkpoint)
    features["lead_speech_ratio"] = round(lead_ratio, 3)
    if lead_ratio < MIN_TURN_SPEECH_RATIO:
        return Candidate(event, features, False, "lead_gates_as_silence")

    ambiguous = last_word is not None and last_word not in CONTINUATION_WORDS
    features["ambiguous_boundary"] = ambiguous
    score = (
        0.6 * min(pause_sec, PAUSE_SEC_CAP) / PAUSE_SEC_CAP
        + 0.2 * min(context_sec, CONTEXT_SEC_CAP) / CONTEXT_SEC_CAP
        + 0.2 * ambiguous
    )
    return Candidate(event, features, True, "selected", round(score, 4))


def evaluate_eot(
    event: SplitEvent,
    window: RolloutWindow,
    labels: dict,
    speaker_paths: dict[str, Path],
    channel_words: dict[str, list[dict]] | None,
) -> Candidate:
    reference = next_turn_after(labels, event.model_speaker, event.timestamp)
    user_turn_sec = event.timestamp - window.turn_start
    reference_sec = reference.end - reference.start
    gap_sec = reference.start - event.timestamp
    reference_words = None
    user_turn_words = None
    if channel_words:
        reference_words = len(
            words_in_span(
                channel_words, event.model_speaker, reference.start, reference.end
            )
        )
        user_turn_words = len(
            words_in_span(
                channel_words, event.active_speaker, window.turn_start, event.timestamp
            )
        )
    features = {
        "user_turn_sec": round(user_turn_sec, 3),
        "user_turn_words": user_turn_words,
        "reference_sec": round(reference_sec, 3),
        "gap_sec": round(gap_sec, 3),
        "reference_word_count": reference_words,
    }
    if user_turn_sec < MIN_USER_TURN_SEC:
        return Candidate(event, features, False, "user_turn_too_short")
    if user_turn_words is not None and user_turn_words < MIN_USER_TURN_WORDS:
        return Candidate(event, features, False, "user_turn_too_few_words")
    if gap_sec > MAX_REFERENCE_GAP_SEC:
        return Candidate(event, features, False, "reference_not_adjacent")
    if reference_words is not None and reference_words < MIN_REFERENCE_WORDS:
        return Candidate(event, features, False, "backchannel_reference")

    reference_ratio = channel_speech_ratio(
        speaker_paths[event.model_speaker], reference.start, reference.end
    )
    features["reference_speech_ratio"] = round(reference_ratio, 3)
    if reference_ratio < MIN_REFERENCE_SPEECH_RATIO:
        return Candidate(event, features, False, "reference_gates_as_silence")
    tail_start = max(window.turn_start, event.timestamp - LEAD_WINDOW_SEC)
    tail_ratio = channel_speech_ratio(
        speaker_paths[event.active_speaker], tail_start, event.timestamp
    )
    features["tail_speech_ratio"] = round(tail_ratio, 3)
    if tail_ratio < MIN_TURN_SPEECH_RATIO:
        return Candidate(event, features, False, "user_tail_gates_as_silence")

    if reference_words is not None:
        content = min(reference_words, REFERENCE_WORDS_CAP) / REFERENCE_WORDS_CAP
    else:
        content = min(reference_sec, REFERENCE_SEC_CAP) / REFERENCE_SEC_CAP
    score = (
        0.6 * content
        + 0.2 * min(user_turn_sec, USER_TURN_SEC_CAP) / USER_TURN_SEC_CAP
        + 0.2 * max(0.0, 1.0 - max(gap_sec, 0.0) / GAP_SEC_CAP)
    )
    return Candidate(event, features, True, "selected", round(score, 4))


def evaluate_event(
    event: SplitEvent,
    labels: dict,
    speaker_paths: dict[str, Path],
    channel_words: dict[str, list[dict]] | None,
) -> Candidate:
    reason = skip_reason(event, labels, speaker_paths)
    if reason is not None:
        return Candidate(event, {}, False, reason)
    window = rollout_window(labels, event)
    if event.split_type == "pause_start":
        return evaluate_pause(event, window, speaker_paths, channel_words)
    return evaluate_eot(event, window, labels, speaker_paths, channel_words)


def select(candidates: list[Candidate], keep_per_type: int) -> None:
    """Rank scored candidates per split type and cut everything past the cap."""
    for split_type in ("pause_start", "end_of_turn"):
        scored = [
            candidate
            for candidate in candidates
            if candidate.event.split_type == split_type
            and candidate.score is not None
        ]
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        for rank, candidate in enumerate(scored):
            if rank < keep_per_type:
                candidate.keep = True
                candidate.reason = "selected"
            else:
                candidate.keep = False
                candidate.reason = "outranked"


def curate_datapoint(
    datapoint_dir: Path,
    channel_words: dict[str, list[dict]] | None,
    keep_per_type: int = DEFAULT_KEEP_PER_TYPE,
) -> list[Candidate]:
    labels = load_labels(datapoint_dir / "label.json")
    speaker_paths = {
        speaker: datapoint_dir / "waves" / f"{speaker}.wav"
        for speaker in ("speaker_1", "speaker_2")
    }
    candidates = [
        evaluate_event(event, labels, speaker_paths, channel_words)
        for event in iter_split_events(labels)
    ]
    select(candidates, keep_per_type)
    return candidates


def curation_payload(
    source_id: str,
    candidates: list[Candidate],
    keep_per_type: int,
) -> dict:
    return {
        "source_id": source_id,
        "keep_per_type": keep_per_type,
        "thresholds": thresholds(),
        "events": [candidate.to_dict() for candidate in candidates],
    }


def write_curation_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def write_curation(
    path: Path,
    source_id: str,
    candidates: list[Candidate],
    keep_per_type: int,
) -> None:
    write_curation_payload(path, curation_payload(source_id, candidates, keep_per_type))


def load_curation(curation_dir: Path, source_id: str) -> dict[int, dict] | None:
    """Decisions for one source keyed by split-event index, or None if uncurated."""
    path = curation_dir / f"{source_id}.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return {event["index"]: event for event in payload["events"]}


def run_curation(
    curation_dir: str | Path,
    datapoints: list[str] | None = None,
    profile: str = "pmt",
    datasource: str = "annotated",
    limit: int | None = None,
    keep_per_type: int = DEFAULT_KEEP_PER_TYPE,
) -> None:
    output_path = Path(curation_dir)
    source = get_source(datasource)
    s3 = boto3.Session(profile_name=profile).client("s3")
    available = source.list_names(s3)
    if datapoints:
        source_names = [resolve_datapoint(name, available) for name in datapoints]
    else:
        source_names = available
    if limit is not None:
        source_names = source_names[:limit]
    if not source_names:
        print(f"No datapoints found for datasource {source.name}")
        return

    def curate_source(temp_dir: str, source_name: str) -> tuple[int, int]:
        datapoint_dir = Path(temp_dir) / source_name.replace("-", "_")
        loaded = source.load(s3, source_name, datapoint_dir)
        candidates = curate_datapoint(
            datapoint_dir, loaded.channel_words, keep_per_type
        )
        shutil.rmtree(datapoint_dir)
        write_curation(
            output_path / f"{loaded.source_id}.json",
            loaded.source_id,
            candidates,
            keep_per_type,
        )
        kept = sum(1 for candidate in candidates if candidate.keep)
        print(
            f"{loaded.source_id}: kept {kept}/{len(candidates)} split events",
            flush=True,
        )
        return len(candidates), kept

    with tempfile.TemporaryDirectory() as temp_dir:
        with ThreadPoolExecutor(max_workers=CURATION_WORKERS) as pool:
            results = list(
                pool.map(lambda name: curate_source(temp_dir, name), source_names)
            )
    total = sum(count for count, _ in results)
    total_kept = sum(kept for _, kept in results)

    figure = write_curation_figures(output_path)
    print(
        f"Done. Kept {total_kept}/{total} split events across "
        f"{len(source_names)} datapoint(s). Figures: {figure}"
    )
