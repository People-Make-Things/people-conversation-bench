"""Tests for speaker persona generation."""

from __future__ import annotations

import json

import pytest

from data_processing.labels import RolloutWindow, SplitEvent
from data_processing.personas import (
    PERSONA_FRAME,
    PROMPT_VERSION,
    compose_text_prompt,
    format_conversation_transcript,
    generate_personas,
    parse_persona_response,
    persona_model_name,
)
from data_processing.prepare_eval import ReferenceResponse, build_point_meta


def labels() -> dict:
    return {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 2.0},
            {"type": "end_of_turn", "timestamp": 3.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 0.5},
            {"type": "end_of_turn", "timestamp": 1.5},
        ],
        "merged": [],
    }


def test_format_conversation_transcript_orders_turns() -> None:
    channel_words = {
        "speaker_1": [{"text": "hello", "start": 2.0, "end": 2.8}],
        "speaker_2": [{"text": "I am from Tehran", "start": 0.5, "end": 1.4}],
    }

    transcript = format_conversation_transcript(labels(), channel_words)

    assert transcript == "speaker_2: I am from Tehran\nspeaker_1: hello"


def test_format_conversation_transcript_skips_empty_turns() -> None:
    channel_words = {
        "speaker_1": [{"text": "hello", "start": 2.0, "end": 2.8}],
        "speaker_2": [],
    }

    transcript = format_conversation_transcript(labels(), channel_words)

    assert transcript == "speaker_1: hello"


def test_compose_text_prompt_always_includes_frame() -> None:
    assert compose_text_prompt("") == PERSONA_FRAME
    assert compose_text_prompt("  ") == PERSONA_FRAME
    composed = compose_text_prompt("You are from Tehran.")
    assert composed.startswith(PERSONA_FRAME)
    assert "You are from Tehran." in composed


def test_parse_persona_response_requires_both_speakers() -> None:
    with pytest.raises(ValueError, match="speaker_2"):
        parse_persona_response('{"speaker_1": "You are brief."}')


def test_generate_personas_wraps_model_output() -> None:
    calls: list[str] = []

    def complete(prompt: str) -> str:
        calls.append(prompt)
        return json.dumps(
            {
                "speaker_1": "You are the cousin visiting from California.",
                "speaker_2": "You grew up in Tehran.",
            }
        )

    payload = generate_personas(
        labels(),
        {
            "speaker_1": [{"text": "hello", "start": 2.0, "end": 2.8}],
            "speaker_2": [{"text": "I am from Tehran", "start": 0.5, "end": 1.4}],
        },
        complete=complete,
    )

    assert len(calls) == 1
    assert "speaker_2: I am from Tehran" in calls[0]
    assert payload["prompt_version"] == PROMPT_VERSION
    assert payload["model"] == persona_model_name()
    assert payload["transcript_sha256"]
    assert payload["speaker_2"]["persona"] == "You grew up in Tehran."
    assert PERSONA_FRAME in payload["speaker_2"]["text_prompt"]
    assert payload["speaker_1"]["text_prompt"].startswith(PERSONA_FRAME)


def test_generate_personas_skips_model_when_transcript_empty() -> None:
    def complete(prompt: str) -> str:
        raise AssertionError(f"should not call model: {prompt}")

    payload = generate_personas(labels(), {}, complete=complete)

    assert payload["model"] is None
    assert payload["transcript_sha256"] is None
    assert payload["speaker_1"]["text_prompt"] == PERSONA_FRAME
    assert payload["speaker_2"]["text_prompt"] == PERSONA_FRAME


def test_build_point_meta_includes_text_prompt() -> None:
    event = SplitEvent(
        index=0,
        timestamp=1.5,
        split_type="pause_start",
        active_speaker="speaker_1",
        model_speaker="speaker_2",
    )
    window = RolloutWindow(
        turn_start=1.0,
        input_end=2.0,
        checkpoint=1.5,
        window_end=2.0,
    )

    meta = build_point_meta(
        "example_000001",
        "example",
        event,
        window,
        24000,
        "rollout_user_input_pause_start.wav",
        compose_text_prompt("You grew up in Tehran."),
    )

    assert meta["assistant_speaker"] == "speaker_2"
    assert meta["text_prompt"].endswith("You grew up in Tehran.")
    assert meta["rollout"]["input_audio"] == "rollout_user_input_pause_start.wav"
    assert "input_audio_pause_end" not in meta["rollout"]


def test_build_point_meta_includes_eot_reference() -> None:
    event = SplitEvent(
        index=0,
        timestamp=3.0,
        split_type="end_of_turn",
        active_speaker="speaker_1",
        model_speaker="speaker_2",
    )
    window = RolloutWindow(
        turn_start=1.0,
        input_end=3.0,
        checkpoint=3.0,
        window_end=3.0,
    )

    meta = build_point_meta(
        "example_000001",
        "example",
        event,
        window,
        24000,
        "rollout_user_input.wav",
        compose_text_prompt("You grew up in Tehran."),
        reference=ReferenceResponse(
            audio="rollout_reference_response.wav",
            transcript="rollout_reference_response_transcript.json",
            start_sec=3.2,
            end_sec=4.7,
            duration_sec=1.5,
            word_count=12,
        ),
    )

    assert meta["split_type"] == "end_of_turn"
    assert meta["rollout"]["expected_action"] == "eot"
    assert meta["rollout"]["reference_audio"] == "rollout_reference_response.wav"
    assert meta["rollout"]["reference_duration_sec"] == 1.5
    assert meta["rollout"]["reference_word_count"] == 12
