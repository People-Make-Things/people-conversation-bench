"""Tests for split-event curation: heuristics, audio gates, and selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bench.audio import write_wav_pcm16
from data_processing.curation import (
    Candidate,
    curate_datapoint,
    load_curation,
    write_curation,
)

SAMPLE_RATE = 24000


def make_session(
    tmp_path: Path,
    s1_events: list[dict],
    s2_events: list[dict],
    s1_speech: list[tuple[float, float]],
    s2_speech: list[tuple[float, float]],
    duration_sec: float = 15.0,
) -> Path:
    root = tmp_path / "session"
    (root / "waves").mkdir(parents=True)
    merged = [
        {"id": speaker, **event}
        for speaker, events in (("speaker_1", s1_events), ("speaker_2", s2_events))
        for event in events
    ]
    merged.sort(key=lambda event: event["timestamp"])
    labels = {"speaker_1": s1_events, "speaker_2": s2_events, "merged": merged}
    (root / "label.json").write_text(json.dumps(labels), encoding="utf-8")

    rng = np.random.default_rng(7)
    for speaker, spans in (("speaker_1", s1_speech), ("speaker_2", s2_speech)):
        pcm = np.zeros(int(duration_sec * SAMPLE_RATE), dtype=np.float32)
        for start, end in spans:
            begin = int(start * SAMPLE_RATE)
            count = int((end - start) * SAMPLE_RATE)
            pcm[begin : begin + count] = rng.normal(0.0, 0.15, count).astype(
                np.float32
            )
        write_wav_pcm16(
            root / "waves" / f"{speaker}.wav", np.clip(pcm, -1.0, 1.0), SAMPLE_RATE
        )
    return root


def event(kind: str, timestamp: float) -> dict:
    return {"type": kind, "timestamp": timestamp}


def base_events(pause_start: float = 5.0, pause_end: float = 6.5) -> tuple[list, list]:
    s1 = [
        event("voice_activation", 0.5),
        event("pause_start", pause_start),
        event("pause_end", pause_end),
        event("end_of_turn", 8.0),
    ]
    s2 = [event("voice_activation", 9.0), event("end_of_turn", 13.0)]
    return s1, s2


def base_words(last_word: str = "week", reference_words: int = 6) -> dict:
    s2_words = [
        {"text": f"word{i}", "start": 9.2 + i * 0.5, "end": 9.5 + i * 0.5}
        for i in range(reference_words)
    ]
    return {
        "speaker_1": [
            {"text": "we", "start": 0.6, "end": 0.8},
            {"text": "left", "start": 1.0, "end": 1.2},
            {"text": last_word, "start": 4.4, "end": 4.9},
        ],
        "speaker_2": s2_words,
    }


def by_type(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    grouped: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.event.split_type, []).append(candidate)
    return grouped


def test_good_pause_and_eot_are_kept(tmp_path: Path) -> None:
    s1, s2 = base_events()
    root = make_session(
        tmp_path, s1, s2, [(0.5, 5.0), (6.5, 8.0)], [(9.0, 13.0)]
    )
    candidates = curate_datapoint(root, base_words())
    grouped = by_type(candidates)

    pause = grouped["pause_start"][0]
    assert pause.keep
    assert pause.reason == "selected"
    assert pause.features["pause_sec"] == 1.5
    assert pause.features["ambiguous_boundary"] is True
    assert pause.score is not None and pause.score > 0.5

    eots = grouped["end_of_turn"]
    assert [c.keep for c in eots] == [True, False]
    assert eots[0].features["reference_word_count"] == 6
    assert eots[1].reason == "no_reference_turn"


def test_pause_after_continuation_word_scores_lower(tmp_path: Path) -> None:
    s1, s2 = base_events()
    root = make_session(
        tmp_path, s1, s2, [(0.5, 5.0), (6.5, 8.0)], [(9.0, 13.0)]
    )
    ambiguous = by_type(curate_datapoint(root, base_words("week")))["pause_start"][0]
    continuation = by_type(curate_datapoint(root, base_words("and")))["pause_start"][0]
    assert continuation.features["ambiguous_boundary"] is False
    assert continuation.score < ambiguous.score


def test_short_pause_is_rejected(tmp_path: Path) -> None:
    s1, s2 = base_events(pause_start=5.0, pause_end=5.3)
    root = make_session(
        tmp_path, s1, s2, [(0.5, 5.0), (5.3, 8.0)], [(9.0, 13.0)]
    )
    pause = by_type(curate_datapoint(root, base_words()))["pause_start"][0]
    assert not pause.keep
    assert pause.reason == "pause_too_short"


def test_pause_with_speech_in_hold_is_rejected(tmp_path: Path) -> None:
    s1, s2 = base_events()
    root = make_session(tmp_path, s1, s2, [(0.5, 8.0)], [(9.0, 13.0)])
    pause = by_type(curate_datapoint(root, base_words()))["pause_start"][0]
    assert not pause.keep
    assert pause.reason == "hold_gates_as_speech"


def test_pause_with_other_speaker_words_in_hold_is_rejected(tmp_path: Path) -> None:
    s1, s2 = base_events()
    root = make_session(
        tmp_path, s1, s2, [(0.5, 5.0), (6.5, 8.0)], [(9.0, 13.0)]
    )
    words = base_words()
    words["speaker_2"].append({"text": "right", "start": 5.5, "end": 5.9})
    pause = by_type(curate_datapoint(root, words))["pause_start"][0]
    assert not pause.keep
    assert pause.reason == "other_speaker_in_hold"


def test_backchannel_reference_is_rejected(tmp_path: Path) -> None:
    s1, s2 = base_events()
    root = make_session(
        tmp_path, s1, s2, [(0.5, 5.0), (6.5, 8.0)], [(9.0, 13.0)]
    )
    eot = by_type(curate_datapoint(root, base_words(reference_words=2)))[
        "end_of_turn"
    ][0]
    assert not eot.keep
    assert eot.reason == "backchannel_reference"


def test_one_word_user_turn_is_rejected(tmp_path: Path) -> None:
    s1, s2 = base_events()
    root = make_session(
        tmp_path, s1, s2, [(0.5, 5.0), (6.5, 8.0)], [(9.0, 13.0)]
    )
    words = base_words()
    words["speaker_1"] = [{"text": "love", "start": 4.4, "end": 4.9}]
    eot = by_type(curate_datapoint(root, words))["end_of_turn"][0]
    assert not eot.keep
    assert eot.reason == "user_turn_too_few_words"


def test_non_adjacent_reference_is_rejected(tmp_path: Path) -> None:
    s1 = [event("voice_activation", 0.5), event("end_of_turn", 3.0)]
    s2 = [event("voice_activation", 8.0), event("end_of_turn", 13.0)]
    root = make_session(tmp_path, s1, s2, [(0.5, 3.0)], [(8.0, 13.0)])
    words = base_words()
    words["speaker_1"] = [
        {"text": f"w{i}", "start": 0.6 + i * 0.5, "end": 0.9 + i * 0.5}
        for i in range(4)
    ]
    words["speaker_2"] = [
        {"text": f"word{i}", "start": 8.2 + i * 0.5, "end": 8.5 + i * 0.5}
        for i in range(6)
    ]
    eot = by_type(curate_datapoint(root, words))["end_of_turn"][0]
    assert not eot.keep
    assert eot.reason == "reference_not_adjacent"
    assert eot.features["gap_sec"] == 5.0


def test_silent_reference_is_rejected(tmp_path: Path) -> None:
    s1, s2 = base_events()
    root = make_session(tmp_path, s1, s2, [(0.5, 5.0), (6.5, 8.0)], [])
    eot = by_type(curate_datapoint(root, base_words()))["end_of_turn"][0]
    assert not eot.keep
    assert eot.reason == "reference_gates_as_silence"


def test_keep_per_type_cap_outranks_lower_scores(tmp_path: Path) -> None:
    s1 = [
        event("voice_activation", 0.5),
        event("pause_start", 3.0),
        event("pause_end", 3.8),
        event("pause_start", 6.0),
        event("pause_end", 8.0),
        event("end_of_turn", 9.5),
    ]
    s2 = [event("voice_activation", 10.0), event("end_of_turn", 13.0)]
    root = make_session(
        tmp_path, s1, s2, [(0.5, 3.0), (3.8, 6.0), (8.0, 9.5)], [(10.0, 13.0)]
    )
    words = base_words()
    words["speaker_1"].append({"text": "office", "start": 5.4, "end": 5.9})
    pauses = by_type(curate_datapoint(root, words, keep_per_type=1))["pause_start"]
    assert [candidate.keep for candidate in pauses] == [False, True]
    assert pauses[0].reason == "outranked"
    assert pauses[1].features["pause_sec"] == 2.0


def test_curation_roundtrip(tmp_path: Path) -> None:
    s1, s2 = base_events()
    root = make_session(
        tmp_path, s1, s2, [(0.5, 5.0), (6.5, 8.0)], [(9.0, 13.0)]
    )
    candidates = curate_datapoint(root, base_words())
    write_curation(tmp_path / "curation" / "session.json", "session", candidates, 3)

    decisions = load_curation(tmp_path / "curation", "session")
    assert decisions is not None
    assert len(decisions) == len(candidates)
    for candidate in candidates:
        decision = decisions[candidate.event.index]
        assert decision["keep"] == candidate.keep
        assert decision["reason"] == candidate.reason
    assert load_curation(tmp_path / "curation", "missing") is None
