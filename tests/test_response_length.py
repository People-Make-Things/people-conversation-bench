from __future__ import annotations

import numpy as np
import pytest

from bench.metrics.length import first_word_playback_sec, score_response_length
from bench.protocol import TimedWord
from bench.speech import PlayedAudio
from bench.trace import RolloutTrace, TraceEvent


def trace(
    expected_action: str,
    audio: list[PlayedAudio],
    events: list[TraceEvent] | None = None,
    timed_out: bool = False,
    text: str = "",
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
        response_audio=np.zeros(0, dtype=np.float32),
        sample_rate=24000,
        timed_out=timed_out,
        played_pcm=[
            np.zeros(int(round((block.end_sec - block.start_sec) * 24000)), dtype=np.float32)
            for block in audio
        ],
    )


def test_response_length_compares_duration_and_words() -> None:
    result = score_response_length(
        trace("eot", [PlayedAudio(1.0, 2.5, 0.1)], text="hello there"),
        human_duration_sec=1.0,
        human_word_count=10,
        model_word_count=15,
        first_word_sec=1.0,
        min_duration_sec=0.05,
    )

    assert result.model_duration_sec == pytest.approx(1.5)
    assert result.duration_delta_sec == pytest.approx(0.5)
    assert result.duration_ratio == pytest.approx(1.5)
    assert result.word_delta == 5
    assert result.word_ratio == pytest.approx(1.5)
    assert result.first_speech_sec == pytest.approx(1.0)
    assert result.last_speech_sec == pytest.approx(2.5)
    assert result.latency_ms == pytest.approx(0.0)
    assert not result.wordless_phonation
    assert not result.empty_response
    assert not result.timed_out


def test_response_length_ignores_speech_before_eot() -> None:
    result = score_response_length(
        trace(
            "eot",
            [
                PlayedAudio(0.2, 0.8, 0.1),
                PlayedAudio(1.2, 2.0, 0.1),
            ],
            text="after the turn",
        ),
        human_duration_sec=2.0,
        human_word_count=8,
        model_word_count=8,
        first_word_sec=1.2,
        min_duration_sec=0.05,
    )

    assert result.model_duration_sec == pytest.approx(0.8)
    assert result.first_speech_sec == pytest.approx(1.2)
    assert result.latency_ms == pytest.approx(200.0)
    assert result.duration_ratio == pytest.approx(0.4)


def test_rollout_with_no_model_audio_is_not_scored() -> None:
    """Nothing came back at all, which is a broken run rather than an
    infinitely terse answer. Averaging it in would drag every ratio to zero."""
    result = score_response_length(
        trace("eot", []),
        human_duration_sec=1.0,
        human_word_count=4,
        model_word_count=0,
        first_word_sec=None,
    )

    assert result.status == "no_model_audio"
    assert result.model_duration_sec == 0.0
    assert result.latency_ms is None
    assert not result.wordless_phonation
    assert result.empty_response


def test_decoded_silence_scores_as_an_empty_response() -> None:
    """The model was alive and generating, it just said nothing. That is a real
    zero-length response, unlike a rollout that returned no audio at all."""
    quiet = [PlayedAudio(1.0 + index * 0.02, 1.02 + index * 0.02, 1e-5) for index in range(50)]
    result = score_response_length(
        trace("eot", quiet),
        human_duration_sec=1.0,
        human_word_count=4,
        model_word_count=0,
        first_word_sec=None,
    )

    assert result.status == "scored"
    assert result.duration_ratio == 0.0
    assert result.word_ratio == 0.0
    assert result.latency_ms is None
    assert not result.wordless_phonation
    assert result.empty_response


def test_response_length_survives_a_quiet_block_mid_word() -> None:
    """Response length uses the same hysteresis gate as pause scoring, so an
    envelope dip inside a word must not truncate the measured response."""
    blocks = [PlayedAudio(1.00, 1.02, 0.05), PlayedAudio(1.02, 1.04, 0.006)]
    blocks += [
        PlayedAudio(1.04 + index * 0.02, 1.06 + index * 0.02, 0.05)
        for index in range(3)
    ]
    result = score_response_length(
        trace("eot", blocks, text="one"),
        human_duration_sec=0.1,
        human_word_count=1,
        model_word_count=1,
        first_word_sec=1.0,
    )

    assert result.first_speech_sec == pytest.approx(1.0)
    assert result.last_speech_sec == pytest.approx(1.10)
    assert result.model_duration_sec == pytest.approx(0.10)


def test_a_rollout_the_drain_cap_cut_off_is_marked_timed_out() -> None:
    """The response was still going when the cap tripped, so its length is a
    lower bound. The score carries that so the batch can keep it out of the mean."""
    result = score_response_length(
        trace("eot", [PlayedAudio(1.0, 30.0, 0.1)], timed_out=True, text="still going"),
        human_duration_sec=4.0,
        human_word_count=12,
        model_word_count=40,
        first_word_sec=1.0,
        min_duration_sec=0.05,
    )

    assert result.timed_out
    assert result.status == "scored"
    assert result.model_duration_sec == pytest.approx(29.0)
    assert result.latency_ms == pytest.approx(0.0)


def test_wordless_murmur_is_not_a_response_onset() -> None:
    """Above-gate audio that transcribes to no words is murmur, not a reply."""
    result = score_response_length(
        trace("eot", [PlayedAudio(1.2, 2.0, 0.1)]),
        human_duration_sec=1.0,
        human_word_count=4,
        model_word_count=0,
        first_word_sec=None,
        min_duration_sec=0.05,
    )

    assert result.status == "scored"
    assert result.first_speech_sec == pytest.approx(1.2)
    assert result.latency_ms is None
    assert result.wordless_phonation


def test_latency_is_measured_to_the_first_word_not_the_murmur() -> None:
    """A model that hums into its first word puts the hum and the word in one
    gate interval. The interval start is the hum; the latency is the word."""
    result = score_response_length(
        trace("eot", [PlayedAudio(1.2, 3.0, 0.1)], text="hello"),
        human_duration_sec=1.0,
        human_word_count=4,
        model_word_count=1,
        first_word_sec=2.0,
        min_duration_sec=0.05,
    )

    assert result.first_speech_sec == pytest.approx(1.2)
    assert result.latency_ms == pytest.approx(1000.0)
    assert not result.wordless_phonation


def test_first_word_playback_time_survives_playback_gaps() -> None:
    """Words are timestamped on the concatenated post-checkpoint clip, which
    squeezed out the wall-clock gap between chunks and the pre-checkpoint
    samples. The mapping must put both back."""
    tr = trace("eot", [PlayedAudio(0.5, 1.25, 0.1), PlayedAudio(3.0, 4.0, 0.1)])
    # checkpoint 1.0: clip is [1.0, 1.25) then [3.0, 4.0)

    assert first_word_playback_sec(tr, [TimedWord("hi", 0.1)]) == pytest.approx(1.1)
    assert first_word_playback_sec(tr, [TimedWord("hi", 0.6)]) == pytest.approx(3.35)
    assert first_word_playback_sec(tr, []) is None
    # Whisper padded its timestamp past the audio it was given.
    assert first_word_playback_sec(tr, [TimedWord("hi", 5.0)]) is None


def test_response_length_rejects_pause_traces() -> None:
    with pytest.raises(ValueError, match="requires eot"):
        score_response_length(
            trace("pause", []),
            human_duration_sec=1.0,
            human_word_count=1,
            model_word_count=1,
            first_word_sec=None,
        )
