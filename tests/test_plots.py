from __future__ import annotations

import json
from pathlib import Path

from bench.metrics.pause import (
    HOLD,
    NO_AUDIO,
    TAKEOVER,
    WORDLESS,
    PauseRecognitionScore,
    pause_outcome,
)
from bench.report.artifacts import load_run_artifacts
from bench.report.plots import (
    LATENCY_NAME,
    LENGTH_AVERAGE_NAME,
    LENGTH_NAME,
    PAUSE_NAME,
    RECALL_NAME,
    SIMILARITY_NAME,
    plot_rows,
    write_run_dir_plots,
)
from bench.report.summarize import INDEX_NAME, write_run_index
from bench.trace import NO_MODEL_AUDIO, SCORED


def write_score(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_write_run_plots_emits_only_present_eval_types(tmp_path: Path) -> None:
    write_score(
        tmp_path / "pause-recognition" / "gpt-realtime-fast" / "example_1_000003" / "rollout_001" / "score.json",
        {
            "eval": "pause-recognition",
            "model": "gpt-realtime-fast",
            "score": 1,
            "false_takeover": False,
            "takeover_latency_ms": None,
            "first_speech_sec": None,
            "labeled_pause_duration_ms": 500.0,
            "model_wait_from_pause_start_ms": None,
            "model_wait_minus_labeled_pause_ms": None,
        },
    )
    write_score(
        tmp_path / "pause-recognition" / "gpt-realtime-fast" / "example_1_000003" / "rollout_002" / "score.json",
        {
            "eval": "pause-recognition",
            "model": "gpt-realtime-fast",
            "score": 0,
            "false_takeover": True,
            "takeover_latency_ms": 120.0,
            "first_speech_sec": 0.12,
            "labeled_pause_duration_ms": 500.0,
            "model_wait_from_pause_start_ms": 120.0,
            "model_wait_minus_labeled_pause_ms": -380.0,
        },
    )
    stale = tmp_path / "pause-recognition" / PAUSE_NAME
    stale.write_bytes(b"stale")

    paths = write_run_dir_plots(tmp_path)

    pause_path = tmp_path / PAUSE_NAME
    model_pause = tmp_path / "pause-recognition" / "gpt-realtime-fast" / PAUSE_NAME
    assert pause_path in paths
    assert model_pause in paths
    assert not stale.exists()
    assert pause_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert model_pause.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (tmp_path / LENGTH_NAME).exists()


def test_write_run_plots_length_figure(tmp_path: Path) -> None:
    write_score(
        tmp_path / "response-length" / "gpt-realtime-fast" / "example_1_000001" / "rollout_001" / "score.json",
        {
            "eval": "response-length",
            "model": "gpt-realtime-fast",
            "human_duration_sec": 1.0,
            "model_duration_sec": 2.0,
            "duration_delta_sec": 1.0,
            "duration_ratio": 2.0,
            "human_word_count": 4,
            "model_word_count": 8,
            "word_delta": 4,
            "word_ratio": 2.0,
            "first_speech_sec": 0.5,
            "last_speech_sec": 2.5,
            "empty_response": False,
            "timed_out": False,
        },
    )

    paths = write_run_dir_plots(tmp_path)

    length_path = tmp_path / LENGTH_NAME
    model_length = tmp_path / "response-length" / "gpt-realtime-fast" / LENGTH_NAME
    assert length_path in paths
    assert model_length in paths
    assert tmp_path / "response-length" / LENGTH_NAME not in paths
    assert length_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert model_length.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (tmp_path / LENGTH_AVERAGE_NAME).exists()
    assert write_run_index(tmp_path) == tmp_path / INDEX_NAME
    assert (tmp_path / INDEX_NAME).is_file()
    assert not (tmp_path / PAUSE_NAME).exists()


def test_write_run_dir_plots_nested_eval_layout(tmp_path: Path) -> None:
    write_score(
        tmp_path
        / "pause-recognition"
        / "gpt-realtime-fast"
        / "example_1_000003"
        / "rollout_001"
        / "score.json",
        {
            "eval": "pause-recognition",
            "model": "gpt-realtime-fast",
            "score": 1,
            "false_takeover": False,
            "takeover_latency_ms": None,
            "first_speech_sec": None,
            "labeled_pause_duration_ms": 500.0,
            "model_wait_from_pause_start_ms": None,
            "model_wait_minus_labeled_pause_ms": None,
        },
    )

    paths = write_run_dir_plots(tmp_path)

    root_pause = tmp_path / PAUSE_NAME
    model_pause = tmp_path / "pause-recognition" / "gpt-realtime-fast" / PAUSE_NAME
    assert root_pause in paths
    assert model_pause in paths
    assert tmp_path / "pause-recognition" / PAUSE_NAME not in paths
    assert root_pause.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_rows_reads_model_from_score_json(tmp_path: Path) -> None:
    write_score(
        tmp_path
        / "pause-recognition"
        / "gpt-realtime-fast"
        / "example_1_000003"
        / "rollout_001"
        / "score.json",
        {
            "eval": "pause-recognition",
            "model": "gpt-realtime-fast",
            "score": 1,
            "false_takeover": False,
            "takeover_latency_ms": None,
            "first_speech_sec": None,
            "labeled_pause_duration_ms": 500.0,
            "model_wait_from_pause_start_ms": None,
            "model_wait_minus_labeled_pause_ms": None,
        },
    )
    write_score(
        tmp_path
        / "response-length"
        / "grok-voice"
        / "example_1_000001"
        / "rollout_001"
        / "score.json",
        {
            "eval": "response-length",
            "model": "grok-voice",
            "human_duration_sec": 1.0,
            "model_duration_sec": 2.0,
            "duration_delta_sec": 1.0,
            "duration_ratio": 2.0,
            "human_word_count": 4,
            "model_word_count": 8,
            "word_delta": 4,
            "word_ratio": 2.0,
            "first_speech_sec": 0.5,
            "last_speech_sec": 2.5,
            "empty_response": False,
            "timed_out": False,
        },
    )

    pause, length, _recall, similarity = plot_rows(load_run_artifacts(tmp_path))

    assert [row.model for row in pause] == ["gpt-realtime-fast"]
    assert [row.model for row in length] == ["grok-voice"]
    assert similarity == []
    write_run_dir_plots(tmp_path)
    assert (tmp_path / PAUSE_NAME).is_file()
    assert (tmp_path / LENGTH_NAME).is_file()


def test_plot_rows_keeps_pause_no_model_audio(tmp_path: Path) -> None:
    write_score(
        tmp_path
        / "pause-recognition"
        / "gpt-realtime-fast"
        / "example_1_000003"
        / "rollout_001"
        / "score.json",
        {
            "eval": "pause-recognition",
            "model": "gpt-realtime-fast",
            "status": "no_model_audio",
            "score": None,
            "false_takeover": False,
            "takeover_latency_ms": None,
            "first_speech_sec": None,
            "labeled_pause_duration_ms": 500.0,
            "model_wait_from_pause_start_ms": None,
            "model_wait_minus_labeled_pause_ms": None,
        },
    )
    write_score(
        tmp_path
        / "pause-recognition"
        / "gpt-realtime-fast"
        / "example_1_000003"
        / "rollout_002"
        / "score.json",
        {
            "eval": "pause-recognition",
            "model": "gpt-realtime-fast",
            "status": "scored",
            "score": 1,
            "false_takeover": False,
            "takeover_latency_ms": None,
            "first_speech_sec": None,
            "labeled_pause_duration_ms": 500.0,
            "model_wait_from_pause_start_ms": None,
            "model_wait_minus_labeled_pause_ms": None,
        },
    )

    pause, _length, _recall, _similarity = plot_rows(
        load_run_artifacts(tmp_path)
    )

    assert [row.score.status for row in pause] == [NO_MODEL_AUDIO, SCORED]
    assert pause_outcome(pause[0].score) == NO_AUDIO
    assert pause_outcome(pause[1].score) == HOLD
    write_run_dir_plots(tmp_path)
    assert (tmp_path / PAUSE_NAME).is_file()


def test_pause_outcome_is_one_vocabulary() -> None:
    dead = PauseRecognitionScore.from_dict(
        {
            "status": "no_model_audio",
            "score": None,
            "false_takeover": False,
            "labeled_pause_duration_ms": 400.0,
            "model_wait_from_pause_start_ms": None,
        }
    )
    hold = PauseRecognitionScore.from_dict(
        {
            "status": "scored",
            "score": 1,
            "false_takeover": False,
            "labeled_pause_duration_ms": 400.0,
        }
    )
    takeover = PauseRecognitionScore.from_dict(
        {
            "status": "scored",
            "score": 0,
            "false_takeover": True,
            "labeled_pause_duration_ms": 400.0,
        }
    )
    wordless = PauseRecognitionScore.from_dict(
        {
            "status": "scored",
            "score": 1,
            "false_takeover": False,
            "labeled_pause_duration_ms": 400.0,
            "speech_ratio": 0.3,
            "wordless_phonation": True,
        }
    )
    assert pause_outcome(dead) == NO_AUDIO
    assert pause_outcome(hold) == HOLD
    assert pause_outcome(takeover) == TAKEOVER
    assert pause_outcome(wordless) == WORDLESS


def test_write_run_dir_plots_writes_per_model_and_bars(tmp_path: Path) -> None:
    for model in ("gpt-realtime-fast", "grok-voice"):
        write_score(
            tmp_path
            / "pause-recognition"
            / model
            / "example_1_000003"
            / "rollout_001"
            / "score.json",
            {
                "eval": "pause-recognition",
                "model": model,
                "score": 1,
                "false_takeover": False,
                "takeover_latency_ms": None,
                "first_speech_sec": None,
                "labeled_pause_duration_ms": 500.0,
                "model_wait_from_pause_start_ms": None,
                "model_wait_minus_labeled_pause_ms": None,
            },
        )
        write_score(
            tmp_path
            / "response-length"
            / model
            / "example_1_000001"
            / "rollout_001"
            / "score.json",
            {
                "eval": "response-length",
                "model": model,
                "human_duration_sec": 1.0,
                "model_duration_sec": 2.0,
                "duration_delta_sec": 1.0,
                "duration_ratio": 2.0,
                "human_word_count": 4,
                "model_word_count": 8,
                "word_delta": 4,
                "word_ratio": 2.0,
                "first_speech_sec": 0.5,
                "last_speech_sec": 2.5,
                "empty_response": False,
                "timed_out": False,
            },
        )

    paths = write_run_dir_plots(tmp_path)

    for model in ("gpt-realtime-fast", "grok-voice"):
        assert tmp_path / "pause-recognition" / model / PAUSE_NAME in paths
        assert tmp_path / "response-length" / model / LENGTH_NAME in paths
    assert (tmp_path / PAUSE_NAME).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert (tmp_path / LENGTH_NAME).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert tmp_path / LENGTH_AVERAGE_NAME in paths
    assert (tmp_path / LENGTH_AVERAGE_NAME).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert write_run_index(tmp_path) == tmp_path / INDEX_NAME
    assert LENGTH_NAME in (tmp_path / INDEX_NAME).read_text(encoding="utf-8")
    assert LENGTH_AVERAGE_NAME in (tmp_path / INDEX_NAME).read_text(encoding="utf-8")


def test_write_run_plots_recall_figure(tmp_path: Path) -> None:
    for eval_type, history, score in (
        ("conversation-recall", 30.0, 1),
        ("conversation-recall", 60.0, 0),
        ("fact-recall", 30.0, 1),
        ("fact-recall", 60.0, 1),
    ):
        write_score(
            tmp_path
            / eval_type
            / "gpt-realtime-fast"
            / f"example_1_{history:.0f}"
            / "rollout_001"
            / "score.json",
            {
                "eval": eval_type,
                "model": "gpt-realtime-fast",
                "status": "scored",
                "score": score,
                "question": "Where from?",
                "gold_answer": "Tehran",
                "transcript": "tehran" if score else "not sure",
                "history_target_sec": history,
                "history_sec": history - 5.0,
                "timed_out": False,
            },
        )

    paths = write_run_dir_plots(tmp_path)

    root = tmp_path / RECALL_NAME
    conversation = tmp_path / "conversation-recall" / "gpt-realtime-fast" / RECALL_NAME
    fact = tmp_path / "fact-recall" / "gpt-realtime-fast" / RECALL_NAME
    assert root in paths
    assert conversation in paths
    assert fact in paths
    assert root.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (tmp_path / "conversation-recall" / RECALL_NAME).exists()
    assert not (tmp_path / PAUSE_NAME).exists()


def similarity_payload(model: str, token_f1: float, cosine: float, words: int) -> dict:
    return {
        "eval": "response-similarity",
        "model": model,
        "token_f1": token_f1,
        "embedding_cosine": cosine,
        "human_text": "hello there",
        "model_text": "hello there",
        "human_word_count": words,
        "model_word_count": words,
        "empty_response": False,
        "timed_out": False,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    }


def test_write_run_plots_similarity_figure(tmp_path: Path) -> None:
    write_score(
        tmp_path
        / "response-length"
        / "gpt-realtime-fast"
        / "example_1_000001"
        / "rollout_001"
        / "similarity.json",
        similarity_payload("gpt-realtime-fast", 0.5, 0.7, 8),
    )

    paths = write_run_dir_plots(tmp_path)

    root = tmp_path / SIMILARITY_NAME
    model_plot = (
        tmp_path / "response-length" / "gpt-realtime-fast" / SIMILARITY_NAME
    )
    assert root in paths
    assert model_plot in paths
    assert root.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert model_plot.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (tmp_path / PAUSE_NAME).exists()
    assert not (tmp_path / LENGTH_NAME).exists()


def test_write_run_plots_similarity_strips_for_multiple_models(tmp_path: Path) -> None:
    for model, f1, cosine in (
        ("gpt-realtime-fast", 0.4, 0.6),
        ("grok-voice", 0.2, 0.3),
    ):
        write_score(
            tmp_path
            / "response-length"
            / model
            / "example_1_000001"
            / "rollout_001"
            / "similarity.json",
            similarity_payload(model, f1, cosine, 4),
        )

    paths = write_run_dir_plots(tmp_path)

    assert tmp_path / SIMILARITY_NAME in paths
    for model in ("gpt-realtime-fast", "grok-voice"):
        assert tmp_path / "response-length" / model / SIMILARITY_NAME in paths
    assert (tmp_path / SIMILARITY_NAME).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_write_run_plots_latency_figure(tmp_path: Path) -> None:
    write_score(
        tmp_path
        / "response-length"
        / "gpt-realtime-fast"
        / "example_1_000001"
        / "rollout_001"
        / "score.json",
        {
            "eval": "response-length",
            "model": "gpt-realtime-fast",
            "human_duration_sec": 1.0,
            "model_duration_sec": 2.0,
            "duration_delta_sec": 1.0,
            "duration_ratio": 2.0,
            "human_word_count": 4,
            "model_word_count": 8,
            "word_delta": 4,
            "word_ratio": 2.0,
            "first_speech_sec": 0.5,
            "last_speech_sec": 2.5,
            "empty_response": False,
            "timed_out": False,
            "latency_ms": 180.0,
            "wordless_phonation": False,
        },
    )
    write_score(
        tmp_path
        / "fact-recall"
        / "grok-voice"
        / "example_1_30"
        / "rollout_001"
        / "score.json",
        {
            "eval": "fact-recall",
            "model": "grok-voice",
            "status": "scored",
            "score": 1,
            "question": "Who?",
            "gold_answer": "Washington",
            "transcript": "washington",
            "history_target_sec": 30.0,
            "history_sec": 25.0,
            "timed_out": False,
            "latency_ms": 320.0,
            "wordless_phonation": False,
        },
    )

    paths = write_run_dir_plots(tmp_path)

    root = tmp_path / LATENCY_NAME
    assert root in paths
    assert root.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert not (tmp_path / "response-latency").exists()
    write_run_index(tmp_path)
    assert LATENCY_NAME not in (tmp_path / INDEX_NAME).read_text(encoding="utf-8")


def test_write_run_plots_does_not_infer_latency_from_trace(tmp_path: Path) -> None:
    rollout = (
        tmp_path
        / "response-length"
        / "gpt-realtime-fast"
        / "example_1_000001"
        / "rollout_001"
    )
    write_score(
        rollout / "score.json",
        {
            "eval": "response-length",
            "model": "gpt-realtime-fast",
            "human_duration_sec": 1.0,
            "model_duration_sec": 2.0,
            "duration_delta_sec": 1.0,
            "duration_ratio": 2.0,
            "human_word_count": 4,
            "model_word_count": 8,
            "word_delta": 4,
            "word_ratio": 2.0,
            "first_speech_sec": 1.2,
            "last_speech_sec": 2.0,
            "empty_response": False,
            "timed_out": False,
        },
    )
    write_score(
        rollout / "trace.json",
        {
            "point_id": "example_1_000001",
            "model_id": "gpt-realtime-fast",
            "rollout_index": 0,
            "expected_action": "eot",
            "checkpoint_sec": 1.0,
            "window_end_sec": 1.0,
            "events": [],
            "played_audio": [{"start_sec": 1.2, "end_sec": 2.0, "rms": 0.1}],
            "text": "hello",
            "sample_rate": 24000,
        },
    )

    assert tmp_path / LATENCY_NAME not in write_run_dir_plots(tmp_path)
