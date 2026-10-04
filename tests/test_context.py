"""Tests for context event generation."""

from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from bench.audio import pcm_float_to_b64, write_wav_pcm16
from data_processing.context import (
    audio_message_event,
    build_context_events,
    build_prefix_context_events,
    extract_turn_text,
    text_message_event,
)
from data_processing.labels import (
    SplitEvent,
    current_turn_start,
    iter_completed_turns,
    iter_turns,
    next_turn_after,
    other_speaker,
    rollout_window,
)


@pytest.fixture
def labels() -> dict:
    return {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 2.0},
            {"type": "end_of_turn", "timestamp": 3.0},
            {"type": "voice_activation", "timestamp": 9.0},
            {"type": "end_of_turn", "timestamp": 10.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 0.5},
            {"type": "end_of_turn", "timestamp": 1.5},
            {"type": "voice_activation", "timestamp": 5.0},
            {"type": "end_of_turn", "timestamp": 6.0},
        ],
        "merged": [],
    }


def test_iter_turns_extracts_voice_activation_to_end_of_turn(labels: dict) -> None:
    turns = iter_turns(labels, "speaker_1")
    assert len(turns) == 2
    assert turns[0].start == 2.0
    assert turns[0].end == 3.0


def test_iter_completed_turns_excludes_active_turn_in_progress(labels: dict) -> None:
    split = SplitEvent(
        index=0,
        timestamp=9.5,
        split_type="pause_start",
        active_speaker="speaker_1",
        model_speaker=other_speaker("speaker_1"),
    )
    completed = iter_completed_turns(labels, split)
    assert [(turn.speaker, turn.start, turn.end) for turn in completed] == [
        ("speaker_2", 0.5, 1.5),
        ("speaker_1", 2.0, 3.0),
        ("speaker_2", 5.0, 6.0),
    ]


def test_iter_completed_turns_excludes_active_end_of_turn_split(labels: dict) -> None:
    split = SplitEvent(
        index=0,
        timestamp=10.0,
        split_type="end_of_turn",
        active_speaker="speaker_1",
        model_speaker=other_speaker("speaker_1"),
    )
    completed = iter_completed_turns(labels, split)
    assert [(turn.speaker, turn.start, turn.end) for turn in completed] == [
        ("speaker_2", 0.5, 1.5),
        ("speaker_1", 2.0, 3.0),
        ("speaker_2", 5.0, 6.0),
    ]


def test_build_prefix_context_events_uses_completed_turns_in_the_window(
    labels: dict,
) -> None:
    events = build_prefix_context_events(
        labels,
        {
            "speaker_1": [{"text": "hello", "start": 2.0, "end": 2.8}],
            "speaker_2": [{"text": "hi", "start": 0.5, "end": 1.4}],
        },
        "speaker_2",
        3.0,
    )

    assert [event["item"]["role"] for event in events] == ["user", "assistant"]
    assert events[0]["item"]["content"][0]["text"] == "hi"
    assert events[1]["item"]["content"][0]["text"] == "hello"


def test_current_turn_start_uses_latest_voice_activation(labels: dict) -> None:
    assert current_turn_start(labels, "speaker_1", 10.0) == 9.0


def test_rollout_window_uses_pause_end_for_pause_window() -> None:
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 1.0},
            {"type": "pause_start", "timestamp": 1.5},
            {"type": "pause_end", "timestamp": 2.0},
            {"type": "end_of_turn", "timestamp": 3.0},
        ],
        "speaker_2": [],
        "merged": [],
    }
    split = SplitEvent(
        index=0,
        timestamp=1.5,
        split_type="pause_start",
        active_speaker="speaker_1",
        model_speaker="speaker_2",
    )

    window = rollout_window(labels, split)

    assert window.turn_start == 1.0
    assert window.checkpoint == 1.5
    assert window.input_end == 2.0
    assert window.window_end == 2.0


def test_rollout_window_for_end_of_turn() -> None:
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 1.0},
            {"type": "end_of_turn", "timestamp": 3.0},
        ],
        "speaker_2": [],
        "merged": [],
    }
    split = SplitEvent(
        index=0,
        timestamp=3.0,
        split_type="end_of_turn",
        active_speaker="speaker_1",
        model_speaker="speaker_2",
    )

    window = rollout_window(labels, split)

    assert window.turn_start == 1.0
    assert window.checkpoint == 3.0
    assert window.input_end == 3.0
    assert window.window_end == 3.0


def test_next_turn_after_returns_first_turn_at_or_after_split(labels: dict) -> None:
    turn = next_turn_after(labels, "speaker_2", 3.0)

    assert turn is not None
    assert turn.start == 5.0
    assert turn.end == 6.0
    assert next_turn_after(labels, "speaker_2", 6.0) is None


def test_extract_turn_text_collects_overlapping_words() -> None:
    words = [
        {"text": "hello", "start": 1.0, "end": 1.5},
        {"text": "world", "start": 1.4, "end": 2.0},
    ]
    assert extract_turn_text(words, 1.0, 2.0) == "hello world"


def test_text_message_event_schema() -> None:
    user_event = text_message_event("user", "hi")
    assistant_event = text_message_event("assistant", "hello")

    assert user_event["type"] == "conversation.item.create"
    assert user_event["item"]["role"] == "user"
    assert user_event["item"]["content"] == [{"type": "input_text", "text": "hi"}]
    assert assistant_event["item"]["content"] == [{"type": "output_text", "text": "hello"}]


def test_audio_message_event_schema() -> None:
    event = audio_message_event("abc123", "partial transcript")
    assert event["item"]["role"] == "user"
    assert event["item"]["content"] == [
        {"type": "input_audio", "audio": "abc123", "transcript": "partial transcript"}
    ]


def test_pcm_float_to_b64_round_trip() -> None:
    pcm = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    encoded = pcm_float_to_b64(pcm)
    raw = base64.b64decode(encoded)
    assert len(raw) == pcm.shape[0] * 2


def test_build_context_events_orders_text_then_final_audio(
    labels: dict,
    tmp_path: Path,
) -> None:
    speaker_paths = {}
    for speaker in ("speaker_1", "speaker_2"):
        path = tmp_path / f"{speaker}.wav"
        pcm = np.linspace(-0.1, 0.1, 24000, dtype=np.float32)
        write_wav_pcm16(path, pcm, 24000)
        speaker_paths[speaker] = path

    channel_words = {
        "speaker_1": [
            {"text": "first", "start": 2.0, "end": 2.8},
            {"text": "second", "start": 9.0, "end": 9.8},
        ],
        "speaker_2": [
            {"text": "reply", "start": 0.5, "end": 1.4},
            {"text": "again", "start": 5.0, "end": 5.8},
        ],
    }
    split = SplitEvent(
        index=0,
        timestamp=9.5,
        split_type="pause_start",
        active_speaker="speaker_1",
        model_speaker="speaker_2",
    )

    events = build_context_events(
        labels,
        channel_words,
        speaker_paths,
        split,
        input_end_sec=10.0,
    )
    assert len(events) == 4
    assert events[0]["item"]["content"][0]["text"] == "reply"
    assert events[1]["item"]["content"][0]["text"] == "first"
    assert events[2]["item"]["content"][0]["text"] == "again"
    assert events[3]["item"]["content"][0]["type"] == "input_audio"
    assert events[3]["item"]["content"][0]["transcript"] == "second"
