"""Tests for eval point context selection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bench.points import EvalPoint, select_context
from bench.protocol import (
    ContextType,
    DuplexAudioContext,
    PreloadNotSupportedError,
    event_context,
)


def test_event_context_classifies_text_vs_text_audio() -> None:
    text_only = event_context(
        [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
    )
    assert text_only.kind is ContextType.TEXT

    text_audio = event_context(
        [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": "abc"}],
                },
            },
        ]
    )
    assert text_audio.kind is ContextType.TEXT_AUDIO


def test_select_context_prefers_model_priority_order() -> None:
    duplex = DuplexAudioContext(
        user_pcm=np.zeros(1920, dtype=np.float32),
        assistant_pcm=np.zeros(1920, dtype=np.float32),
        sample_rate=24000,
    )
    events = event_context(
        [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
    )
    point = EvalPoint(
        point_id="test",
        point_dir=Path("."),
        meta={},
        contexts={
            ContextType.DUPLEX_AUDIO: duplex,
            ContextType.TEXT: events,
        },
    )

    selected = select_context(
        point.contexts, (ContextType.TEXT, ContextType.DUPLEX_AUDIO)
    )
    assert selected is events

    selected = select_context(
        point.contexts, (ContextType.DUPLEX_AUDIO, ContextType.TEXT)
    )
    assert selected is duplex


def test_select_context_raises_when_no_supported_kind() -> None:
    events = event_context(
        [
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
        ]
    )
    point = EvalPoint(
        point_id="test",
        point_dir=Path("."),
        meta={},
        contexts={ContextType.TEXT: events},
    )

    with pytest.raises(PreloadNotSupportedError, match="duplex_audio"):
        select_context(point.contexts, (ContextType.DUPLEX_AUDIO,))
