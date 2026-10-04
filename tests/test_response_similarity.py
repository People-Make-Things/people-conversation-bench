from __future__ import annotations

import numpy as np
import pytest

from bench.metrics.similarity import (
    cosine_similarity,
    score_response_similarity,
    token_f1,
)
from bench.speech import PlayedAudio
from bench.trace import RolloutTrace


def trace(
    expected_action: str,
    timed_out: bool = False,
    played: bool = True,
) -> RolloutTrace:
    return RolloutTrace(
        point_id="point",
        model_id="model",
        rollout_index=0,
        expected_action=expected_action,
        checkpoint_sec=1.0,
        window_end_sec=1.0,
        events=[],
        played_audio=[PlayedAudio(1.0, 2.0, 0.1)] if played else [],
        text="",
        response_audio=np.zeros(0, dtype=np.float32),
        sample_rate=24000,
        timed_out=timed_out,
    )


def unit_embed(texts: list[str]) -> np.ndarray:
    table = {
        "yeah that makes sense": np.array([1.0, 0.0], dtype=np.float32),
        "right i get it": np.array([0.8, 0.6], dtype=np.float32),
        "the weather is terrible": np.array([0.0, 1.0], dtype=np.float32),
    }
    return np.stack([table.get(text, np.ones(2, dtype=np.float32)) for text in texts])


def test_token_f1_is_one_for_the_same_words() -> None:
    assert token_f1("Yeah, that makes sense.", "yeah that makes sense") == 1.0


def test_token_f1_is_zero_when_either_side_is_empty() -> None:
    assert token_f1("hello", "") == 0.0
    assert token_f1("", "hello") == 0.0
    assert token_f1("", "") == 0.0


def test_token_f1_is_zero_with_no_overlap() -> None:
    assert token_f1("hello there", "goodbye now") == 0.0


def test_token_f1_partial_overlap() -> None:
    assert token_f1("hello there friend", "hello friend") == pytest.approx(0.8)


def test_cosine_similarity_is_one_for_parallel_vectors() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([2.0, 0.0])) == pytest.approx(
        1.0
    )


def test_cosine_similarity_is_zero_for_empty_or_orthogonal() -> None:
    assert cosine_similarity(np.zeros(2), np.array([1.0, 0.0])) == 0.0
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(
        0.0
    )


def test_score_keeps_both_texts_and_both_metrics() -> None:
    result = score_response_similarity(
        trace("eot"),
        human_text="yeah that makes sense",
        model_text="right i get it",
        embed=unit_embed,
    )

    assert result.token_f1 == 0.0
    assert result.embedding_cosine == pytest.approx(0.8)
    assert result.human_text == "yeah that makes sense"
    assert result.model_text == "right i get it"
    assert result.human_word_count == 4
    assert result.model_word_count == 4
    assert not result.empty_response
    assert not result.timed_out


def test_empty_model_text_is_an_empty_response() -> None:
    result = score_response_similarity(
        trace("eot"),
        human_text="hello there",
        model_text="",
        embed=unit_embed,
    )

    assert result.token_f1 == 0.0
    assert result.embedding_cosine == 0.0
    assert result.empty_response


def test_rollout_with_no_model_audio_is_not_scored() -> None:
    result = score_response_similarity(
        trace("eot", played=False),
        human_text="hello there",
        model_text="",
        embed=unit_embed,
    )

    assert result.status == "no_model_audio"
    assert result.empty_response


def test_a_rollout_the_drain_cap_cut_off_is_marked_timed_out() -> None:
    result = score_response_similarity(
        trace("eot", timed_out=True),
        human_text="hello there",
        model_text="hello there",
        embed=unit_embed,
    )

    assert result.timed_out
    assert result.status == "scored"
    assert result.token_f1 == 1.0


def test_response_similarity_rejects_pause_traces() -> None:
    with pytest.raises(ValueError, match="requires eot"):
        score_response_similarity(
            trace("pause"),
            human_text="hello",
            model_text="hello",
            embed=unit_embed,
        )
