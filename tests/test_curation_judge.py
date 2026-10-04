"""Tests for judge verdict scoring, re-selection, and window rendering."""

from __future__ import annotations

from data_processing.curation_judge import (
    JUDGE_VERSION,
    eot_verdict,
    needs_judging,
    pause_verdict,
    render_window,
    reselect,
    scale_int,
)


def test_scale_int_coerces_and_clamps() -> None:
    assert scale_int(7) == 7
    assert scale_int("7") == 7
    assert scale_int(7.6) == 7
    assert scale_int(None) == 0
    assert scale_int(15) == 10
    assert scale_int(-2) == 0


def test_pause_verdict_scores_only_fair_candidates() -> None:
    fair = pause_verdict({"coherent": True, "turn_handoff": False, "temptation": 8})
    assert fair["score"] == 0.8
    handoff = pause_verdict({"coherent": True, "turn_handoff": True, "temptation": 9})
    assert handoff["score"] == 0.0
    incoherent = pause_verdict(
        {"coherent": False, "turn_handoff": False, "temptation": 9}
    )
    assert incoherent["score"] == 0.0


def test_eot_verdict_requires_all_flags() -> None:
    good = eot_verdict(
        {
            "coherent": True,
            "genuine_handoff": True,
            "responsive_reference": True,
            "difficulty": 6,
        }
    )
    assert good["score"] == 0.6
    fragment = eot_verdict(
        {
            "coherent": True,
            "genuine_handoff": False,
            "responsive_reference": True,
            "difficulty": 9,
        }
    )
    assert fragment["score"] == 0.0


def survivor(index: int, split_type: str, score: float, judge_score) -> dict:
    event = {
        "index": index,
        "split_type": split_type,
        "score": score,
        "keep": False,
        "reason": "outranked",
    }
    if judge_score is not None:
        event["judge"] = {"version": JUDGE_VERSION, "model": "m", "score": judge_score}
    return event


def test_reselect_ranks_by_judge_with_quality_floor() -> None:
    payload = {
        "keep_per_type": 2,
        "events": [
            survivor(0, "pause_start", 0.9, 0.0),
            survivor(1, "pause_start", 0.2, 0.8),
            survivor(2, "pause_start", 0.5, 0.4),
            survivor(3, "pause_start", 0.4, 0.4),
            survivor(4, "pause_start", 0.9, 0.3),
        ],
    }
    reselect(payload)
    events = {event["index"]: event for event in payload["events"]}
    assert not events[0]["keep"] and events[0]["reason"] == "judge_rejected"
    assert events[1]["keep"]
    # judge tie at 0.4 broken by heuristic score
    assert events[2]["keep"]
    assert not events[3]["keep"] and events[3]["reason"] == "outranked"
    # below MIN_JUDGE_SCORE is rejected even though a cap slot is free
    assert not events[4]["keep"] and events[4]["reason"] == "judge_rejected"


def test_needs_judging_on_version_and_model() -> None:
    gated_out = {"score": None}
    assert not needs_judging(gated_out, "m")
    fresh = survivor(0, "pause_start", 0.5, None)
    assert needs_judging(fresh, "m")
    judged = survivor(1, "pause_start", 0.5, 0.4)
    assert not needs_judging(judged, "m")
    assert needs_judging(judged, "other-model")


def test_render_pause_window_includes_boundary_words() -> None:
    event = {
        "split_type": "pause_start",
        "timestamp": 10.0,
        "active_speaker": "speaker_1",
        "features": {"pause_sec": 1.5, "context_sec": 5.0},
    }
    words = {
        "speaker_1": [
            {"text": "we", "start": 5.5, "end": 5.8},
            {"text": "left", "start": 6.0, "end": 6.3},
            {"text": "early", "start": 9.0, "end": 9.5},
            {"text": "anyway", "start": 12.0, "end": 12.4},
        ],
        "speaker_2": [{"text": "right", "start": 4.0, "end": 4.3}],
    }
    window = render_window(event, words)
    assert 'LAST WORDS BEFORE THE SILENCE: "we left early"' in window
    assert "SILENT FOR 1.5 SECONDS" in window
    assert "anyway" in window


def test_render_eot_window_labels_reference() -> None:
    event = {
        "split_type": "end_of_turn",
        "timestamp": 10.0,
        "active_speaker": "speaker_2",
        "features": {
            "user_turn_sec": 4.0,
            "reference_sec": 3.0,
            "gap_sec": 0.5,
        },
    }
    words = {
        "speaker_2": [{"text": "question", "start": 7.0, "end": 7.4}],
        "speaker_1": [
            {"text": "before", "start": 2.0, "end": 2.4},
            {"text": "answer", "start": 11.0, "end": 11.5},
        ],
    }
    window = render_window(event, words)
    assert "USER TURN (ends here, 4.0s): question" in window
    assert "ASSISTANT REPLY (human reference, 3.0s after 0.5s gap): answer" in window
