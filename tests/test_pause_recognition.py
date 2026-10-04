from __future__ import annotations

import numpy as np
import pytest

from bench.metrics.pause import PauseRecognitionScore, score_pause_recognition
from bench.speech import PlayedAudio
from bench.trace import RolloutTrace, TraceEvent


def trace(
    expected_action: str,
    audio: list[PlayedAudio],
    events: list[TraceEvent] | None = None,
    text: str = "",
    response_sec: float = 0.0,
) -> RolloutTrace:
    return RolloutTrace(
        point_id="point",
        model_id="model",
        rollout_index=0,
        expected_action=expected_action,
        checkpoint_sec=1.0,
        window_end_sec=2.0,
        events=events or [],
        played_audio=audio,
        text=text,
        response_audio=np.zeros(int(response_sec * 24000), dtype=np.float32),
        sample_rate=24000,
    )


def test_pause_fails_when_model_speaks_during_pause() -> None:
    result = score_pause_recognition(
        trace("pause", [PlayedAudio(1.2, 1.4, 0.1)], text="hello"),
        min_duration_sec=0.05,
    )

    assert result.score == 0
    assert result.false_takeover
    assert result.takeover_latency_ms == pytest.approx(200)
    assert result.labeled_pause_duration_ms == pytest.approx(1000)
    assert result.model_wait_from_pause_start_ms == pytest.approx(200)
    assert result.model_wait_minus_labeled_pause_ms == pytest.approx(-800)


def test_pause_passes_when_model_waits_for_resume() -> None:
    result = score_pause_recognition(
        trace("pause", [PlayedAudio(2.0, 2.2, 0.1)], text="okay"),
        min_duration_sec=0.05,
    )

    assert result.score == 1
    assert not result.false_takeover
    assert result.takeover_latency_ms is None
    assert result.labeled_pause_duration_ms == pytest.approx(1000)
    assert result.model_wait_from_pause_start_ms == pytest.approx(1000)
    assert result.model_wait_minus_labeled_pause_ms == pytest.approx(0)


def test_rollout_with_no_model_audio_is_not_scored() -> None:
    """A rollout where nothing came back is a broken run, not a perfect pause.

    It used to score 1, which is how a dead socket, a model that was never
    stepped, and a genuinely patient model all became the same number."""
    result = score_pause_recognition(trace("pause", []))

    assert result.status == "no_model_audio"
    assert result.score is None
    assert result.labeled_pause_duration_ms == pytest.approx(1000)
    assert result.model_wait_from_pause_start_ms is None
    assert result.model_wait_minus_labeled_pause_ms is None


def test_decoded_silence_still_scores_because_the_model_was_generating() -> None:
    quiet = [PlayedAudio(index * 0.02, index * 0.02 + 0.02, 1e-5) for index in range(150)]
    result = score_pause_recognition(trace("pause", quiet))

    assert result.status == "scored"
    assert result.score == 1
    assert result.model_audio_sec == 0.0


def test_speech_survives_a_single_quiet_block_mid_word() -> None:
    """Hysteresis: an envelope dip inside a word must not end the interval and
    leave two fragments short enough to discard."""
    blocks = [PlayedAudio(1.00, 1.02, 0.05), PlayedAudio(1.02, 1.04, 0.006)]
    blocks += [
        PlayedAudio(1.04 + index * 0.02, 1.06 + index * 0.02, 0.05)
        for index in range(3)
    ]
    result = score_pause_recognition(
        trace("pause", blocks, text="hello"),
        min_duration_sec=0.08,
    )

    assert result.score == 0
    assert result.false_takeover
    assert result.rms_threshold == pytest.approx(0.01)


def test_speech_carried_in_from_before_the_checkpoint_is_flagged_separately() -> None:
    blocks = [
        PlayedAudio(0.80 + index * 0.02, 0.82 + index * 0.02, 0.05)
        for index in range(20)
    ]
    result = score_pause_recognition(
        trace("pause", blocks, text="hello"),
        min_duration_sec=0.05,
    )

    assert result.speech_continued_from_before_checkpoint
    assert result.false_takeover


def test_score_records_the_whole_speech_gate() -> None:
    """The gate has three calibrated parameters, so recording only the threshold
    left a score that could not be reproduced from its own artifact."""
    result = score_pause_recognition(
        trace("pause", [PlayedAudio(1.2, 1.4, 0.1)]),
        rms_threshold=0.02,
        min_duration_sec=0.05,
        release_ratio=0.75,
    )

    assert result.to_dict()["rms_threshold"] == pytest.approx(0.02)
    assert result.to_dict()["release_ratio"] == pytest.approx(0.75)
    assert result.to_dict()["min_duration_sec"] == pytest.approx(0.05)
    assert PauseRecognitionScore.from_dict(result.to_dict()) == result


def test_the_gate_opens_on_the_block_where_speech_starts() -> None:
    """Onset fidelity is what the calibration bought and what the pause score
    turns on: a late onset converts a real takeover into a false pass."""
    blocks = [PlayedAudio(index * 0.02, 0.02 + index * 0.02, 1e-5) for index in range(50)]
    blocks += [
        PlayedAudio(1.0 + index * 0.02, 1.02 + index * 0.02, 0.05) for index in range(10)
    ]

    result = score_pause_recognition(trace("pause", blocks, text="hello"))

    assert result.first_speech_sec == pytest.approx(1.0)
    assert result.takeover_latency_ms == pytest.approx(0.0)


def test_wordless_audio_during_the_hold_is_not_a_takeover() -> None:
    """Above-gate murmur is not uttered words, so it is not a response."""
    blocks = [
        PlayedAudio(1.2 + index * 0.02, 1.22 + index * 0.02, 0.05) for index in range(10)
    ]

    result = score_pause_recognition(trace("pause", blocks, response_sec=2.0))

    assert result.score == 1
    assert not result.false_takeover
    assert result.model_wait_from_pause_start_ms is None
    assert result.wordless_phonation


def test_a_pass_carries_whether_it_was_earned_by_wordless_phonation() -> None:
    blocks = [
        PlayedAudio(2.0 + index * 0.02, 2.02 + index * 0.02, 0.05) for index in range(50)
    ]

    wordless = score_pause_recognition(trace("pause", blocks, response_sec=4.0))
    spoken = score_pause_recognition(
        trace("pause", blocks, text="hello there", response_sec=4.0)
    )

    assert wordless.score == 1 and spoken.score == 1
    assert wordless.speech_ratio == pytest.approx(0.25)
    assert wordless.wordless_phonation
    assert not spoken.wordless_phonation


def test_pause_scoring_rejects_an_eot_trace() -> None:
    with pytest.raises(ValueError):
        score_pause_recognition(trace("eot", []))
