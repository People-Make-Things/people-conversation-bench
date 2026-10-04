"""Tests for lazy eval point loading."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bench.audio import write_wav_pcm16
from bench.history import condition_duplex_history
from bench.points import (
    EotScenario,
    PauseScenario,
    load_point,
    load_rollout_scenario,
    normalize_expected_action,
    point_text_prompt,
)
from bench.protocol import ContextType, DuplexAudioContext, TimedWord
from utils.timed_words import write_timed_words


def test_load_point_skips_duplex_audio_when_not_requested(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "context": "context.json",
                "user_audio": "user.wav",
                "assistant_audio": "assistant.wav",
            }
        ),
        encoding="utf-8",
    )
    (point_dir / "context.json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "hello"}],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    point = load_point(point_dir, kinds=(ContextType.TEXT, ContextType.TEXT_AUDIO))

    assert ContextType.TEXT in point.contexts
    assert ContextType.DUPLEX_AUDIO not in point.contexts


def test_load_point_requires_wav_files_for_duplex(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "user_audio": "user.wav",
                "assistant_audio": "assistant.wav",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError):
        load_point(point_dir, kinds=(ContextType.DUPLEX_AUDIO,))


def test_load_rollout_scenario_uses_completed_context_and_input(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "sample_rate": 24000,
                "text_prompt": "Stay in character.",
                "rollout": {
                    "expected_action": "pause",
                    "history_end_sec": 1.0,
                    "checkpoint_sec": 0.25,
                    "window_end_sec": 0.5,
                    "context": "rollout_context.json",
                    "input_audio": "rollout_user_input_pause_start.wav",
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
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "hello"}],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    # Pause eval streams through pause_start; window_end may extend past the WAV.
    write_wav_pcm16(
        point_dir / "rollout_user_input_pause_start.wav",
        np.zeros(6000, dtype=np.float32),
        24000,
    )
    write_wav_pcm16(
        point_dir / "rollout_user_input_pause_end.wav",
        np.zeros(12000, dtype=np.float32),
        24000,
    )

    scenario = load_rollout_scenario(
        point_dir,
        kinds=(ContextType.TEXT, ContextType.TEXT_AUDIO),
    )

    assert scenario.expected_action == "pause"
    assert scenario.checkpoint_sec == 0.25
    assert scenario.window_end_sec == 0.5
    assert scenario.input_pcm.shape == (6000,)
    assert scenario.text_prompt == "Stay in character."
    assert ContextType.TEXT in scenario.contexts


def test_load_rollout_scenario_rejects_unknown_action(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "sample_rate": 24000,
                "rollout": {
                    "expected_action": "hold",
                    "history_end_sec": 1.0,
                    "checkpoint_sec": 0.25,
                    "window_end_sec": 0.25,
                    "input_audio": "rollout_user_input_pause_start.wav",
                },
            }
        ),
        encoding="utf-8",
    )
    write_wav_pcm16(
        point_dir / "rollout_user_input_pause_start.wav",
        np.zeros(6000, dtype=np.float32),
        24000,
    )
    write_wav_pcm16(
        point_dir / "rollout_user_input_pause_end.wav",
        np.zeros(6000, dtype=np.float32),
        24000,
    )

    with pytest.raises(ValueError, match="unknown expected action"):
        load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))


def test_load_rollout_scenario_rejects_checkpoint_past_input(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "sample_rate": 24000,
                "rollout": {
                    "expected_action": "pause",
                    "checkpoint_sec": 0.5,
                    "window_end_sec": 0.75,
                    "input_audio": "rollout_user_input_pause_start.wav",
                },
            }
        ),
        encoding="utf-8",
    )
    write_wav_pcm16(
        point_dir / "rollout_user_input_pause_start.wav",
        np.zeros(6000, dtype=np.float32),
        24000,
    )

    with pytest.raises(ValueError, match="checkpoint is outside rollout input"):
        load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))


def write_pause_point(
    point_dir: Path,
    checkpoint_sec: float,
    window_end_sec: float,
    pause_end_sec: float | None,
) -> None:
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "sample_rate": 24000,
                "rollout": {
                    "expected_action": "pause",
                    "checkpoint_sec": checkpoint_sec,
                    "window_end_sec": window_end_sec,
                    "input_audio": "rollout_user_input_pause_start.wav",
                },
            }
        ),
        encoding="utf-8",
    )
    write_wav_pcm16(
        point_dir / "rollout_user_input_pause_start.wav",
        np.zeros(int(checkpoint_sec * 24000), dtype=np.float32),
        24000,
    )
    if pause_end_sec is not None:
        write_wav_pcm16(
            point_dir / "rollout_user_input_pause_end.wav",
            np.zeros(int(pause_end_sec * 24000), dtype=np.float32),
            24000,
        )


def test_load_rollout_scenario_picks_up_pause_end_audio(tmp_path: Path) -> None:
    """Duplex rollouts stream this to keep the model stepping through the pause."""
    point_dir = tmp_path / "000001"
    write_pause_point(point_dir, checkpoint_sec=0.24, window_end_sec=0.8, pause_end_sec=0.8)

    scenario = load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))

    assert scenario.input_pcm.shape == (int(0.24 * 24000),)
    assert isinstance(scenario, PauseScenario)
    assert scenario.pause_input_pcm.shape == (int(0.8 * 24000),)


def test_load_rollout_scenario_requires_pause_end_audio(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    write_pause_point(point_dir, checkpoint_sec=0.24, window_end_sec=0.8, pause_end_sec=None)

    with pytest.raises(ValueError, match="missing"):
        load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))


def test_load_rollout_scenario_rejects_input_running_past_the_checkpoint(
    tmp_path: Path,
) -> None:
    """The streamed input has to end at the checkpoint, or the model hears
    speech the scoring window assumes it never got."""
    point_dir = tmp_path / "000001"
    write_pause_point(point_dir, checkpoint_sec=0.24, window_end_sec=0.8, pause_end_sec=0.8)
    write_wav_pcm16(
        point_dir / "rollout_user_input_pause_start.wav",
        np.zeros(int(0.6 * 24000), dtype=np.float32),
        24000,
    )

    with pytest.raises(ValueError, match="past the checkpoint"):
        load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))


def test_load_rollout_scenario_rejects_window_past_pause_end_audio(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    write_pause_point(point_dir, checkpoint_sec=0.24, window_end_sec=1.5, pause_end_sec=0.8)

    with pytest.raises(ValueError, match="ends after"):
        load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))


def test_load_rollout_scenario_requires_eot_reference(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "sample_rate": 24000,
                "rollout": {
                    "expected_action": "eot",
                    "checkpoint_sec": 0.25,
                    "window_end_sec": 0.25,
                    "input_audio": "rollout_user_input.wav",
                },
            }
        ),
        encoding="utf-8",
    )
    write_wav_pcm16(
        point_dir / "rollout_user_input.wav",
        np.zeros(6000, dtype=np.float32),
        24000,
    )

    with pytest.raises(ValueError, match="missing reference response"):
        load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))


def test_load_rollout_scenario_requires_reference_transcript(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "sample_rate": 24000,
                "rollout": {
                    "expected_action": "eot",
                    "checkpoint_sec": 0.25,
                    "window_end_sec": 0.25,
                    "input_audio": "rollout_user_input.wav",
                    "reference_duration_sec": 1.5,
                    "reference_word_count": 12,
                },
            }
        ),
        encoding="utf-8",
    )
    write_wav_pcm16(
        point_dir / "rollout_user_input.wav",
        np.zeros(6000, dtype=np.float32),
        24000,
    )

    with pytest.raises(ValueError, match="missing reference transcript"):
        load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))


def test_load_rollout_scenario_reads_reference_transcript(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "sample_rate": 24000,
                "rollout": {
                    "expected_action": "eot",
                    "checkpoint_sec": 0.25,
                    "window_end_sec": 0.25,
                    "input_audio": "rollout_user_input.wav",
                    "reference_duration_sec": 1.5,
                    "reference_word_count": 2,
                    "reference_transcript": "rollout_reference_response_transcript.json",
                },
            }
        ),
        encoding="utf-8",
    )
    write_wav_pcm16(
        point_dir / "rollout_user_input.wav",
        np.zeros(6000, dtype=np.float32),
        24000,
    )
    write_timed_words(
        point_dir / "rollout_reference_response_transcript.json",
        (
            TimedWord(text="hello", start_sec=0.0, end_sec=0.3),
            TimedWord(text="there", start_sec=0.3, end_sec=0.6),
        ),
    )

    scenario = load_rollout_scenario(point_dir, kinds=(ContextType.TEXT,))

    assert isinstance(scenario, EotScenario)
    assert scenario.reference.duration_sec == 1.5
    assert scenario.reference.word_count == 2
    assert scenario.reference.text == "hello there"


def test_normalize_expected_action_accepts_eot() -> None:
    assert normalize_expected_action("eot") == "eot"


def test_normalize_expected_action_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown expected action"):
        normalize_expected_action("interrupt")


def duplex_scenario(frames: int) -> PauseScenario:
    """A pause scenario whose duplex history is `frames` 80 ms frames long, with
    one word landing in each second of history."""
    samples = frames * 1920
    duration_sec = samples / 24000
    return PauseScenario(
        point_id="example_000001",
        point_dir=Path("."),
        contexts={
            ContextType.DUPLEX_AUDIO: DuplexAudioContext(
                user_pcm=np.arange(samples, dtype=np.float32),
                assistant_pcm=np.zeros(samples, dtype=np.float32),
                sample_rate=24000,
                words=tuple(
                    TimedWord(text=f"w{index}", start_sec=index, end_sec=index + 0.5)
                    for index in range(int(duration_sec))
                ),
            )
        },
        input_pcm=np.zeros(1920, dtype=np.float32),
        pause_input_pcm=np.zeros(1920, dtype=np.float32),
        sample_rate=24000,
        checkpoint_sec=0.08,
        window_end_sec=0.08,
    )


def duplex_context(scenario: PauseScenario) -> DuplexAudioContext:
    context = scenario.contexts[ContextType.DUPLEX_AUDIO]
    assert isinstance(context, DuplexAudioContext)
    return context


def test_condition_duplex_history_is_a_no_op_by_default() -> None:
    scenario = duplex_scenario(frames=50)

    assert condition_duplex_history(scenario) is scenario


def test_condition_duplex_history_keeps_the_most_recent_audio() -> None:
    """Trimming drops the oldest frames, the same end the server truncates, and
    rebases the word stream so prefill still teacher-forces aligned text."""
    scenario = duplex_scenario(frames=50)

    trimmed = duplex_context(condition_duplex_history(scenario, history_sec=1.2))

    assert trimmed.user_pcm.shape == (15 * 1920,)
    assert trimmed.assistant_pcm.shape == trimmed.user_pcm.shape
    original = duplex_context(scenario).user_pcm
    assert np.array_equal(trimmed.user_pcm, original[-15 * 1920 :])
    # 4 s of history became its last 1.2 s, so only the word at 3.0 s survives,
    # 0.2 s from the new start.
    assert [word.text for word in trimmed.words] == ["w3"]
    assert trimmed.words[0].start_sec == pytest.approx(0.2)


def test_condition_duplex_history_keeps_at_least_one_frame() -> None:
    scenario = duplex_scenario(frames=50)

    trimmed = duplex_context(condition_duplex_history(scenario, history_sec=0.0))

    assert trimmed.user_pcm.shape == (1920,)
    assert trimmed.words == ()


def test_condition_duplex_history_can_drop_the_teacher_forced_text() -> None:
    """Same audio, no word stream, so prefill forces all PAD instead of the
    aligned assistant transcript."""
    scenario = duplex_scenario(frames=50)

    conditioned = duplex_context(
        condition_duplex_history(scenario, teacher_forced_text=False)
    )

    assert conditioned.words == ()
    assert np.array_equal(conditioned.user_pcm, duplex_context(scenario).user_pcm)


def test_load_point_reads_text_prompt(tmp_path: Path) -> None:
    point_dir = tmp_path / "000001"
    point_dir.mkdir()
    (point_dir / "point.json").write_text(
        json.dumps(
            {
                "id": "example_000001",
                "text_prompt": "Stay in character.",
                "context": "context.json",
            }
        ),
        encoding="utf-8",
    )
    (point_dir / "context.json").write_text(
        json.dumps({"events": []}),
        encoding="utf-8",
    )

    point = load_point(point_dir, kinds=(ContextType.TEXT, ContextType.TEXT_AUDIO))

    assert point.text_prompt == "Stay in character."
    assert point_text_prompt({"text_prompt": "  "}) is None
