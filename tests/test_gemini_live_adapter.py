"""Tests for Gemini Live adapter setup, VAD, and prefill."""

from __future__ import annotations

import asyncio
import json
import sys

import numpy as np

import pytest

from bench.audio import pcm_float_to_b64
from bench.protocol import (
    AudioFrame,
    AudioInterrupted,
    ContextType,
    SessionConfig,
    SessionMode,
    event_context,
)
from bench.registry import load_adapter, load_model_config, resolve_model_ref


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        pass


class EventWebSocket(FakeWebSocket):
    def __init__(self, events: list[dict]) -> None:
        super().__init__()
        self.events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return json.dumps(next(self.events))
        except StopIteration:
            raise StopAsyncIteration


def adapter_module(model):
    return sys.modules[model.__class__.__module__]


def loaded_setup_payload():
    config = load_model_config(resolve_model_ref("gemini-live"))
    model = load_adapter(config)
    return config, adapter_module(model).setup_payload


def test_preload_context_converts_gpt_events_to_client_content() -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gemini-live"))
        model = load_adapter(config)
        model._ws = FakeWebSocket()
        context = event_context(
            [
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    },
                },
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

        await model.preload_context(context)

        assert context.kind is ContextType.TEXT_AUDIO
        payload = json.loads(model._ws.sent[0])["clientContent"]
        assert payload["turnComplete"] is True
        assert payload["turns"][0] == {
            "role": "model",
            "parts": [{"text": "hello"}],
        }
        assert payload["turns"][1]["role"] == "user"
        assert payload["turns"][1]["parts"][0]["inlineData"]["data"] == "abc"

    asyncio.run(run())


def test_setup_payload_live_vad() -> None:
    config, setup_payload = loaded_setup_payload()
    setup = setup_payload(config.session, SessionMode.LIVE)["setup"]
    assert setup["model"] == "models/gemini-3.1-flash-live-preview"
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["generationConfig"]["speechConfig"]["voiceConfig"][
        "prebuiltVoiceConfig"
    ]["voiceName"] == "Kore"
    assert setup["realtimeInputConfig"] == {
        "automaticActivityDetection": {
            "disabled": False,
            "prefixPaddingMs": 300,
            "silenceDurationMs": 500,
        },
        "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
    }
    assert "historyConfig" not in setup


def test_setup_payload_preload_enables_initial_history() -> None:
    config, setup_payload = loaded_setup_payload()
    setup = setup_payload(config.session, SessionMode.PRELOAD)["setup"]
    assert setup["historyConfig"]["initialHistoryInClientContent"] is True


def test_setup_payload_eot_disables_automatic_vad() -> None:
    config, setup_payload = loaded_setup_payload()
    session = SessionConfig(detect_turns=False, extra=dict(config.session.extra))
    setup = setup_payload(session, SessionMode.LIVE)["setup"]
    assert setup["realtimeInputConfig"] == {
        "automaticActivityDetection": {"disabled": True},
    }


def test_connect_waits_for_setup_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gemini-live"))
        model = load_adapter(config)
        websocket = EventWebSocket([{"setupComplete": {}}])
        module = adapter_module(model)

        async def connect(*args, **kwargs):
            return websocket

        monkeypatch.setattr(module, "require_api_key", lambda config: "test-key")
        monkeypatch.setattr(module.websockets, "connect", connect)

        await model.connect(config.session)
        assert json.loads(websocket.sent[0])["setup"]["model"].endswith(
            "gemini-3.1-flash-live-preview"
        )
        await model.close()

    asyncio.run(run())


def test_end_input_closes_manual_activity() -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gemini-live"))
        model = load_adapter(config)
        websocket = FakeWebSocket()
        model._ws = websocket
        model._manual_turns = True

        await model.send_audio(np.zeros(480, dtype=np.float32))
        await model.end_input()

        kinds = [json.loads(item)["realtimeInput"] for item in websocket.sent]
        assert kinds[0] == {"activityStart": {}}
        assert "audio" in kinds[1]
        assert kinds[2] == {"activityEnd": {}}

    asyncio.run(run())


def test_interrupted_is_exposed_as_audio_event() -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gemini-live"))
        model = load_adapter(config)

        await model.dispatch_event(
            {"serverContent": {"interrupted": True, "turnComplete": True}}
        )

        event = await anext(model.recv_audio())
        signal = await anext(model.recv_signals())
        assert isinstance(event, AudioInterrupted)
        assert signal.kind == "interrupted"

    asyncio.run(run())


def test_model_audio_and_transcript_are_queued() -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gemini-live"))
        model = load_adapter(config)
        pcm = np.zeros(8, dtype=np.float32)

        await model.dispatch_event(
            {
                "serverContent": {
                    "modelTurn": {
                        "parts": [
                            {
                                "inlineData": {
                                    "mimeType": "audio/pcm;rate=24000",
                                    "data": pcm_float_to_b64(pcm),
                                }
                            }
                        ]
                    },
                    "outputTranscription": {"text": "hi"},
                }
            }
        )

        frame = await anext(model.recv_audio())
        text = await anext(model.recv_text())
        assert isinstance(frame, AudioFrame)
        assert frame.pcm.shape[0] == 8
        assert text == "hi"

    asyncio.run(run())


def test_missing_live_model_fails() -> None:
    _, setup_payload = loaded_setup_payload()
    with pytest.raises(SystemExit, match="live_model"):
        setup_payload(SessionConfig(detect_turns=False), SessionMode.LIVE)
