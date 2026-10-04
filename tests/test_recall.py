from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.audio import write_wav_pcm16
from bench.metrics.recall import (
    CONVERSATION_JUDGE_SYSTEM,
    JUDGE_SYSTEM,
    rescore_recall_payload,
    rescore_recall_tree,
    score_recall,
)
from bench.points import (
    RecallScenario,
    conversation_text,
    eval_type_for_point,
    load_rollout_scenario,
)
from bench.protocol import ContextType, SessionConfig
from bench.report.summarize import recall_results
from bench.rollout import apply_session_config
from bench.speech import PlayedAudio
from bench.trace import NO_MODEL_AUDIO, SCORED, RolloutTrace
from data_processing.cuts import recall_history_cuts
from data_processing.labels import iter_prefix_turns, prefix_end_sec
from data_processing.recall import (
    RECALL_ASSISTANT,
    RECALL_USER,
    extract_conversation_fact,
    fact_for_source,
    judge_conversation_fact,
    style_probe,
    write_recall_points,
)


def recall_labels() -> dict:
    return {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 25.0},
            {"type": "end_of_turn", "timestamp": 35.0},
            {"type": "voice_activation", "timestamp": 45.0},
            {"type": "end_of_turn", "timestamp": 70.0},
            {"type": "voice_activation", "timestamp": 95.0},
            {"type": "end_of_turn", "timestamp": 125.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 25.0},
            {"type": "voice_activation", "timestamp": 35.0},
            {"type": "end_of_turn", "timestamp": 45.0},
            {"type": "voice_activation", "timestamp": 70.0},
            {"type": "end_of_turn", "timestamp": 95.0},
        ],
        "merged": [],
    }


def test_prefix_end_uses_last_completed_turn_at_or_before_target() -> None:
    labels = recall_labels()

    assert prefix_end_sec(labels, 30.0) == 25.0
    assert prefix_end_sec(labels, 60.0) == 45.0
    assert prefix_end_sec(labels, 90.0) == 70.0
    assert prefix_end_sec(labels, 120.0) == 95.0
    assert prefix_end_sec(labels, 10.0) is None


def test_iter_prefix_turns_includes_both_speakers() -> None:
    turns = iter_prefix_turns(recall_labels(), 45.0)

    assert [(turn.speaker, turn.end) for turn in turns] == [
        ("speaker_2", 25.0),
        ("speaker_1", 35.0),
        ("speaker_2", 45.0),
    ]


def idle_recall_labels() -> dict:
    """Handoffs sit in long idle gaps so 30/60/90/120 are clean cuts."""
    return {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 31.0},
            {"type": "end_of_turn", "timestamp": 40.0},
            {"type": "voice_activation", "timestamp": 61.0},
            {"type": "end_of_turn", "timestamp": 75.0},
            {"type": "voice_activation", "timestamp": 121.0},
            {"type": "end_of_turn", "timestamp": 130.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 20.0},
            {"type": "voice_activation", "timestamp": 40.0},
            {"type": "end_of_turn", "timestamp": 50.0},
            {"type": "voice_activation", "timestamp": 91.0},
            {"type": "end_of_turn", "timestamp": 100.0},
        ],
        "merged": [],
    }


def test_history_cuts_skip_targets_far_from_a_clean_cut() -> None:
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 40.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 40.0},
            {"type": "end_of_turn", "timestamp": 55.0},
        ],
        "merged": [],
    }

    assert recall_history_cuts(labels) == [(60.0, 55.0)]


def test_history_cuts_snap_to_silence_inside_the_slack_window() -> None:
    assert recall_history_cuts(idle_recall_labels()) == [
        (30.0, 30.0),
        (60.0, 60.0),
        (90.0, 90.0),
        (120.0, 120.0),
    ]


def test_fact_for_source_is_stable() -> None:
    assert fact_for_source("example_1") == fact_for_source("example_1")


def test_extract_conversation_fact_returns_none_when_empty() -> None:
    def complete(_system: str, _prompt: str) -> str:
        return json.dumps({"question": "", "answer": ""})

    assert extract_conversation_fact("speaker_2: hey", complete=complete) is None


def test_style_probe_falls_back_to_the_question() -> None:
    def complete(_system: str, _prompt: str) -> str:
        return json.dumps({"text": ""})

    assert style_probe("Who left?", "yeah", complete=complete) == "Who left?"


def write_silence(path: Path, duration_sec: float) -> None:
    write_wav_pcm16(
        path,
        np.zeros(int(duration_sec * 24000), dtype=np.float32),
        24000,
    )


def stub_complete(system: str, prompt: str) -> str:
    if "extract one factual question" in system:
        return json.dumps(
            {"question": "Where did they grow up?", "answer": "Tehran"}
        )
    if "fair memory probe" in system:
        return json.dumps({"keep": 1, "reason": "specific place"})
    return json.dumps({"text": "wait, where did they grow up again?"})


def test_write_recall_points_reuses_one_probe_per_metric(tmp_path: Path) -> None:
    datapoint = tmp_path / "source"
    waves = datapoint / "waves"
    waves.mkdir(parents=True)
    write_silence(waves / "speaker_1.wav", 130.0)
    write_silence(waves / "speaker_2.wav", 130.0)
    points_dir = tmp_path / "points"
    points_dir.mkdir()
    spoken: list[str] = []
    transcribed: list[Path] = []

    def synthesize(_pcm, _rate, _ref_text, prompt_text: str):
        spoken.append(prompt_text)
        return np.zeros(1920, dtype=np.float32)

    def transcribe(path: Path):
        transcribed.append(path)
        return []

    paths = write_recall_points(
        source_id="example_1",
        labels=idle_recall_labels(),
        speaker_paths={
            "speaker_1": waves / "speaker_1.wav",
            "speaker_2": waves / "speaker_2.wav",
        },
        points_dir=points_dir,
        personas_payload={
            RECALL_ASSISTANT: {"text_prompt": "Stay in character."},
            RECALL_USER: {"text_prompt": "unused"},
        },
        channel_words={
            "speaker_1": [{"text": "I grew up in Tehran", "start": 26.0, "end": 28.0}],
            "speaker_2": [{"text": "oh nice", "start": 1.0, "end": 2.0}],
        },
        synthesize=synthesize,
        complete=stub_complete,
        transcribe=transcribe,
    )

    assert len(paths) == 8
    assert len(transcribed) == 4
    assert spoken == [
        "wait, where did they grow up again?",
        "wait, where did they grow up again?",
    ]
    actions = []
    targets = []
    for path in paths:
        meta = json.loads((tmp_path / path).read_text(encoding="utf-8"))
        actions.append(meta["rollout"]["expected_action"])
        targets.append(meta["rollout"]["history_target_sec"])
        assert meta["user_speaker"] == "speaker_2"
        assert meta["assistant_speaker"] == "speaker_1"
        assert meta["text_prompt"] == "Stay in character."
        if meta["rollout"]["expected_action"] == "fact_recall":
            assert meta["rollout"]["gold_answer"]
        else:
            assert "gold_answer" not in meta["rollout"]
        assert (tmp_path / path).parent.joinpath("rollout_user_input.wav").is_file()
        assert not (tmp_path / path).parent.joinpath("user.wav").exists()
    assert actions.count("fact_recall") == 4
    assert actions.count("conversation_recall") == 4
    assert targets == [30.0, 30.0, 60.0, 60.0, 90.0, 90.0, 120.0, 120.0]


def test_judge_conversation_fact_parses_verdict() -> None:
    def accept(_system: str, _prompt: str) -> str:
        return json.dumps({"keep": 1, "reason": "names a city"})

    def reject(_system: str, _prompt: str) -> str:
        return json.dumps({"keep": 0, "reason": "generic introduction"})

    assert judge_conversation_fact("t", "q", "a", complete=accept) == (
        True,
        "names a city",
    )
    assert judge_conversation_fact("t", "q", "a", complete=reject) == (
        False,
        "generic introduction",
    )


def test_rejected_fact_drops_conversation_recall(tmp_path: Path) -> None:
    datapoint = tmp_path / "source"
    waves = datapoint / "waves"
    waves.mkdir(parents=True)
    write_silence(waves / "speaker_1.wav", 130.0)
    write_silence(waves / "speaker_2.wav", 130.0)
    points_dir = tmp_path / "points"
    points_dir.mkdir()

    def rejecting_complete(system: str, prompt: str) -> str:
        if "fair memory probe" in system:
            return json.dumps({"keep": 0, "reason": "too generic"})
        return stub_complete(system, prompt)

    paths = write_recall_points(
        source_id="example_1",
        labels=idle_recall_labels(),
        speaker_paths={
            "speaker_1": waves / "speaker_1.wav",
            "speaker_2": waves / "speaker_2.wav",
        },
        points_dir=points_dir,
        personas_payload={
            RECALL_ASSISTANT: {"text_prompt": "Stay in character."},
            RECALL_USER: {"text_prompt": "unused"},
        },
        channel_words={
            "speaker_1": [{"text": "I grew up in Tehran", "start": 26.0, "end": 28.0}],
            "speaker_2": [{"text": "oh nice", "start": 1.0, "end": 2.0}],
        },
        synthesize=lambda *_args: np.zeros(1920, dtype=np.float32),
        complete=rejecting_complete,
        transcribe=lambda _path: [],
    )

    actions = [
        json.loads((tmp_path / path).read_text(encoding="utf-8"))["rollout"][
            "expected_action"
        ]
        for path in paths
    ]
    assert actions == ["fact_recall"] * 4


def test_probe_leaking_speaker_label_drops_conversation_recall(
    tmp_path: Path,
) -> None:
    datapoint = tmp_path / "source"
    waves = datapoint / "waves"
    waves.mkdir(parents=True)
    write_silence(waves / "speaker_1.wav", 130.0)
    write_silence(waves / "speaker_2.wav", 130.0)
    points_dir = tmp_path / "points"
    points_dir.mkdir()

    def leaking_complete(system: str, prompt: str) -> str:
        if "rewrite a question" in system and "grow up" in prompt:
            return json.dumps({"text": "so what does speaker_2 regret?"})
        return stub_complete(system, prompt)

    paths = write_recall_points(
        source_id="example_1",
        labels=idle_recall_labels(),
        speaker_paths={
            "speaker_1": waves / "speaker_1.wav",
            "speaker_2": waves / "speaker_2.wav",
        },
        points_dir=points_dir,
        personas_payload={
            RECALL_ASSISTANT: {"text_prompt": "Stay in character."},
            RECALL_USER: {"text_prompt": "unused"},
        },
        channel_words={
            "speaker_1": [{"text": "I grew up in Tehran", "start": 26.0, "end": 28.0}],
            "speaker_2": [{"text": "oh nice", "start": 1.0, "end": 2.0}],
        },
        synthesize=lambda *_args: np.zeros(1920, dtype=np.float32),
        complete=leaking_complete,
        transcribe=lambda _path: [],
    )

    actions = [
        json.loads((tmp_path / path).read_text(encoding="utf-8"))["rollout"][
            "expected_action"
        ]
        for path in paths
    ]
    assert actions == ["fact_recall"] * 4


def test_missing_tts_dependency_raises(tmp_path: Path) -> None:
    datapoint = tmp_path / "source"
    waves = datapoint / "waves"
    waves.mkdir(parents=True)
    write_silence(waves / "speaker_1.wav", 130.0)
    write_silence(waves / "speaker_2.wav", 130.0)
    points_dir = tmp_path / "points"
    points_dir.mkdir()

    def broken_synthesize(*_args):
        raise RuntimeError("voice cloning requires qwen-tts")

    with pytest.raises(RuntimeError, match="qwen-tts"):
        write_recall_points(
            source_id="example_1",
            labels=recall_labels(),
            speaker_paths={
                "speaker_1": waves / "speaker_1.wav",
                "speaker_2": waves / "speaker_2.wav",
            },
            points_dir=points_dir,
            personas_payload={
                RECALL_ASSISTANT: {"text_prompt": "x"},
                RECALL_USER: {"text_prompt": "y"},
            },
            channel_words=None,
            synthesize=broken_synthesize,
            complete=stub_complete,
            transcribe=lambda _path: [],
        )


def test_write_recall_points_skips_short_user_speech(tmp_path: Path) -> None:
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 40.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 40.0},
            {"type": "end_of_turn", "timestamp": 41.0},
        ],
        "merged": [],
    }
    paths = write_recall_points(
        source_id="short",
        labels=labels,
        speaker_paths={"speaker_1": tmp_path / "a.wav", "speaker_2": tmp_path / "b.wav"},
        points_dir=tmp_path,
        personas_payload={
            RECALL_ASSISTANT: {"text_prompt": "x"},
            RECALL_USER: {"text_prompt": "y"},
        },
        channel_words=None,
        start_index=3,
        synthesize=lambda *_args: np.zeros(1920, dtype=np.float32),
        complete=stub_complete,
        transcribe=lambda _path: [],
    )

    assert paths == []


def test_eval_type_for_recall_points() -> None:
    assert (
        eval_type_for_point({"rollout": {"expected_action": "conversation_recall"}})
        == "conversation-recall"
    )
    assert (
        eval_type_for_point({"rollout": {"expected_action": "fact_recall"}})
        == "fact-recall"
    )


def test_load_rollout_scenario_reads_recall_fields(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "sample_rate": 24000,
                "text_prompt": "Stay in character.",
                "rollout": {
                    "expected_action": "conversation_recall",
                    "checkpoint_sec": 0.08,
                    "window_end_sec": 0.08,
                    "history_target_sec": 30.0,
                    "history_sec": 25.0,
                    "question": "wait, where did they grow up?",
                    "context": "rollout_context.json",
                    "input_audio": "rollout_user_input.wav",
                },
            }
        ),
        encoding="utf-8",
    )
    (point_dir / "rollout_context.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "I grew up in Tehran"}
                            ],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    write_wav_pcm16(
        point_dir / "rollout_user_input.wav",
        np.zeros(1920, dtype=np.float32),
        24000,
    )

    scenario = load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))

    assert isinstance(scenario, RecallScenario)
    assert scenario.expected_action == "conversation_recall"
    assert scenario.question == "wait, where did they grow up?"
    assert scenario.gold_answer == ""
    assert scenario.conversation == "user: I grew up in Tehran"
    assert scenario.history_target_sec == 30.0
    assert scenario.history_sec == 25.0


def recall_trace(
    action: str, audio: list[PlayedAudio], text: str = ""
) -> RolloutTrace:
    return RolloutTrace(
        point_id="point",
        model_id="model",
        rollout_index=0,
        expected_action=action,
        checkpoint_sec=1.0,
        window_end_sec=1.0,
        events=[],
        played_audio=audio,
        text=text,
        response_audio=np.zeros(0, dtype=np.float32),
        sample_rate=24000,
    )


def test_conversation_text_keeps_roles() -> None:
    text = conversation_text(
        [
            {
                "item": {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "I grew up in Tehran"}],
                }
            },
            {
                "item": {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "oh nice"}],
                }
            },
        ]
    )

    assert text == "user: I grew up in Tehran\nassistant: oh nice"


def test_conversation_recall_judge_sees_history(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def complete(system: str, prompt: str, *, model: str) -> str:
        seen["system"] = system
        seen["prompt"] = prompt
        seen["model"] = model
        return json.dumps({"correct": True})

    monkeypatch.setattr("bench.metrics.recall.openai_json_complete", complete)
    result = score_recall(
        recall_trace("conversation_recall", [PlayedAudio(1.0, 2.0, 0.1)]),
        question="wait, where did they grow up?",
        history_target_sec=30.0,
        history_sec=25.0,
        first_word_sec=None,
        transcript="they grew up in Tehran",
        conversation="user: I grew up in Tehran\nassistant: oh nice",
    )

    assert result.score == 1
    assert result.conversation == "user: I grew up in Tehran\nassistant: oh nice"
    assert seen["system"] == CONVERSATION_JUDGE_SYSTEM
    assert "I grew up in Tehran" in seen["prompt"]
    assert "Gold answer" not in seen["prompt"]


def test_fact_recall_judge_uses_gold(monkeypatch) -> None:
    seen: dict[str, str] = {}

    def complete(system: str, prompt: str, *, model: str) -> str:
        seen["system"] = system
        seen["prompt"] = prompt
        return json.dumps({"correct": True})

    monkeypatch.setattr("bench.metrics.recall.openai_json_complete", complete)
    result = score_recall(
        recall_trace("fact_recall", [PlayedAudio(1.0, 2.0, 0.1)]),
        question="Who was first?",
        gold_answer="George Washington",
        history_target_sec=60.0,
        history_sec=45.0,
        first_word_sec=None,
        transcript="George Washington",
    )

    assert result.score == 1
    assert seen["system"] == JUDGE_SYSTEM
    assert "Gold answer: George Washington" in seen["prompt"]
    assert "Conversation:" not in seen["prompt"]


def test_conversation_recall_requires_history() -> None:
    with pytest.raises(ValueError, match="requires conversation history"):
        score_recall(
            recall_trace("conversation_recall", [PlayedAudio(1.0, 2.0, 0.1)]),
            question="Where?",
            history_target_sec=30.0,
            history_sec=25.0,
            first_word_sec=None,
            transcript="Tehran",
        )


def test_score_recall_is_binary(monkeypatch) -> None:
    monkeypatch.setattr("bench.metrics.recall.judge_correct", lambda *_args, **_kwargs: True)
    result = score_recall(
        recall_trace("conversation_recall", [PlayedAudio(1.0, 2.0, 0.1)]),
        question="Where?",
        history_target_sec=30.0,
        history_sec=25.0,
        first_word_sec=None,
        transcript="they grew up in Tehran",
        conversation="user: I grew up in Tehran",
    )

    assert result.eval == "conversation-recall"
    assert result.score == 1
    assert result.status == SCORED
    assert "gold_answer" not in result.to_dict()


def test_score_recall_miss_is_zero(monkeypatch) -> None:
    monkeypatch.setattr("bench.metrics.recall.judge_correct", lambda *_args, **_kwargs: False)
    result = score_recall(
        recall_trace("fact_recall", [PlayedAudio(1.0, 2.0, 0.1)]),
        question="Who was first?",
        gold_answer="George Washington",
        history_target_sec=60.0,
        history_sec=45.0,
        first_word_sec=None,
        transcript="I am not sure",
    )

    assert result.eval == "fact-recall"
    assert result.score == 0


def test_recall_with_no_model_audio_is_not_scored() -> None:
    result = score_recall(
        recall_trace("fact_recall", []),
        question="Who?",
        gold_answer="Washington",
        history_target_sec=30.0,
        history_sec=25.0,
        first_word_sec=None,
        transcript="",
    )

    assert result.status == NO_MODEL_AUDIO
    assert result.score is None
    assert result.latency_ms is None
    assert not result.wordless_phonation


def test_recall_latency_is_wait_until_words(monkeypatch) -> None:
    monkeypatch.setattr("bench.metrics.recall.judge_correct", lambda *_args, **_kwargs: True)
    result = score_recall(
        recall_trace(
            "fact_recall",
            [PlayedAudio(1.2, 2.0, 0.1)],
            text="George Washington",
        ),
        question="Who was first?",
        gold_answer="George Washington",
        history_target_sec=60.0,
        history_sec=45.0,
        transcript="George Washington",
        first_word_sec=1.2,
    )

    assert result.latency_ms == pytest.approx(200.0)
    assert not result.wordless_phonation


def test_recall_wordless_murmur_is_not_a_response_onset(monkeypatch) -> None:
    monkeypatch.setattr("bench.metrics.recall.judge_correct", lambda *_args, **_kwargs: False)
    result = score_recall(
        recall_trace("fact_recall", [PlayedAudio(1.2, 2.0, 0.1)]),
        question="Who was first?",
        gold_answer="George Washington",
        history_target_sec=60.0,
        history_sec=45.0,
        first_word_sec=None,
        transcript="",
    )

    assert result.latency_ms is None
    assert result.wordless_phonation


def test_recall_results_group_by_history_length(monkeypatch) -> None:
    answers = iter((True, False, True))
    monkeypatch.setattr(
        "bench.metrics.recall.judge_correct",
        lambda *_args, **_kwargs: next(answers),
    )
    scores = [
        score_recall(
            recall_trace("fact_recall", [PlayedAudio(1.0, 2.0, 0.1)]),
            question="q",
            gold_answer="a",
            history_target_sec=target,
            history_sec=target,
            first_word_sec=None,
            transcript="ok",
        )
        for target in (30.0, 30.0, 60.0)
    ]

    results = recall_results(scores, [], [])

    assert results["rollout_count"] == 3
    assert results["mean_score"] == pytest.approx(2 / 3)
    assert results["by_history_sec"] == {"30": 0.5, "60": 1.0}


def test_recall_uses_the_eot_drain() -> None:
    scenario = RecallScenario(
        point_id="p",
        point_dir=Path("."),
        contexts={},
        input_pcm=np.zeros(1920, dtype=np.float32),
        sample_rate=24000,
        checkpoint_sec=0.08,
        window_end_sec=0.08,
        expected_action="fact_recall",
        question="q",
        gold_answer="a",
        history_target_sec=30.0,
        history_sec=25.0,
    )
    session = SessionConfig(detect_turns=True)

    # Both families share this path, so one assertion covers duplex and hosted.
    assert apply_session_config(session, scenario).detect_turns is False


def stored_recall(eval_name: str, **overrides) -> dict:
    data = {
        "eval": eval_name,
        "status": "scored",
        "score": 0,
        "question": "where?",
        "transcript": "Tehran",
        "history_target_sec": 30.0,
        "history_sec": 25.0,
        "timed_out": False,
        "model": "gpt-realtime-fast",
    }
    if eval_name == "conversation-recall":
        data["conversation"] = "user: Tehran"
    else:
        data["gold_answer"] = "Tehran"
    data.update(overrides)
    return data


def test_rescore_recall_payload_records_the_new_judge() -> None:
    seen: dict[str, str] = {}

    def complete(system: str, prompt: str, *, model: str) -> str:
        seen["system"] = system
        seen["prompt"] = prompt
        seen["model"] = model
        return json.dumps({"correct": True})

    payload = rescore_recall_payload(
        stored_recall("conversation-recall"),
        model="claude-opus-5-5",
        complete=complete,
    )

    assert payload["score"] == 1
    assert payload["judge_model"] == "claude-opus-5-5"
    assert seen["model"] == "claude-opus-5-5"
    assert seen["system"] == CONVERSATION_JUDGE_SYSTEM
    assert "user: Tehran" in seen["prompt"]
    assert "Gold answer" not in seen["prompt"]


def test_rescore_fact_recall_sends_the_gold_answer() -> None:
    seen: dict[str, str] = {}

    def complete(system: str, prompt: str, *, model: str) -> str:
        seen["system"] = system
        seen["prompt"] = prompt
        return json.dumps({"correct": True})

    rescore_recall_payload(
        stored_recall("fact-recall"),
        model="claude-opus-5-5",
        complete=complete,
    )

    assert seen["system"] == JUDGE_SYSTEM
    assert "Gold answer: Tehran" in seen["prompt"]


def test_rescore_leaves_unscored_rollouts_unchanged() -> None:
    def complete(*_args, **_kwargs) -> str:
        raise AssertionError("judge called")

    original = stored_recall("fact-recall", status="no_model_audio", score=None)

    assert (
        rescore_recall_payload(
            original, model="claude-opus-5-5", complete=complete
        )
        == original
    )


def test_rescore_recall_tree_writes_a_parallel_run(tmp_path: Path) -> None:
    def complete(_system: str, _prompt: str, *, model: str) -> str:
        assert model == "claude-opus-5-5"
        return json.dumps({"correct": False})

    source = tmp_path / "run"
    scored = (
        source
        / "conversation-recall"
        / "gpt-realtime-fast"
        / "p"
        / "rollout_000"
    )
    quiet = source / "fact-recall" / "moshi" / "p" / "rollout_000"
    failed = source / "fact-recall" / "moshi" / "q" / "rollout_000"
    pause = source / "pause-recognition" / "gpt-realtime-fast" / "p" / "rollout_000"
    for path in (scored, quiet, failed, pause):
        path.mkdir(parents=True)
    (scored / "score.json").write_text(
        json.dumps(stored_recall("conversation-recall", score=1)),
        encoding="utf-8",
    )
    (quiet / "score.json").write_text(
        json.dumps(stored_recall("fact-recall", status="no_model_audio", score=None)),
        encoding="utf-8",
    )
    (failed / "error.json").write_text('{"eval": "fact-recall"}\n', encoding="utf-8")
    (pause / "score.json").write_text("{}", encoding="utf-8")

    output = tmp_path / "out"
    counts = rescore_recall_tree(
        source,
        output,
        model="claude-opus-5-5",
        complete=complete,
        workers=2,
    )

    assert counts == {"judged": 1, "copied": 2, "skipped": 0}
    written = json.loads(
        (
            output
            / "conversation-recall"
            / "gpt-realtime-fast"
            / "p"
            / "rollout_000"
            / "score.json"
        ).read_text(encoding="utf-8")
    )
    assert written["score"] == 0
    assert written["judge_model"] == "claude-opus-5-5"
    assert json.loads((scored / "score.json").read_text(encoding="utf-8"))["score"] == 1
    assert not (output / "pause-recognition").exists()
    assert (output / "fact-recall" / "moshi" / "q" / "rollout_000" / "error.json").is_file()

    again = rescore_recall_tree(
        source,
        output,
        model="claude-opus-5-5",
        complete=complete,
        workers=2,
    )
    assert again == {"judged": 0, "copied": 0, "skipped": 3}
