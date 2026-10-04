"""Tests for GPT Realtime adapter prefill."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from bench.protocol import ContextType, SessionConfig, event_context
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


def test_preload_context_only_sends_history_items() -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gpt-realtime-fast"))
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
        assert len(model._ws.sent) == 2
        assert json.loads(model._ws.sent[0])["item"]["role"] == "assistant"
        assert json.loads(model._ws.sent[1])["item"]["content"][0]["type"] == "input_audio"

    asyncio.run(run())


def test_speech_does_not_interrupt_when_disabled() -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gpt-realtime-fast"))
        model = load_adapter(config)
        model._ws = FakeWebSocket()
        model._interrupt_response = False

        await model.dispatch_event({"type": "input_audio_buffer.speech_started"})

        assert model._audio_queue.empty()

    asyncio.run(run())


def test_initial_speech_does_not_emit_playback_interruption() -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gpt-realtime-fast"))
        model = load_adapter(config)
        model._ws = FakeWebSocket()

        await model.dispatch_event({"type": "input_audio_buffer.speech_started"})

        assert model._audio_queue.empty()

    asyncio.run(run())


def test_speech_stopped_is_exposed_as_model_signal() -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gpt-realtime-fast"))
        model = load_adapter(config)

        await model.dispatch_event(
            {
                "type": "input_audio_buffer.speech_stopped",
                "item_id": "item",
                "audio_end_ms": 1200,
            }
        )

        signal = await anext(model.recv_signals())
        assert signal.kind == "input_speech_stopped"
        assert signal.data["audio_end_ms"] == 1200

    asyncio.run(run())


@pytest.mark.parametrize(
    ("model_id", "silence_duration_ms"),
    (("gpt-realtime-fast", 300), ("gpt-realtime-slow", 500)),
)
def test_vad_profiles_send_explicit_session_settings(
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
    silence_duration_ms: int,
) -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref(model_id))
        model = load_adapter(config)
        websocket = EventWebSocket([{"type": "session.updated"}])
        module = sys.modules[model.__class__.__module__]

        async def connect(*args, **kwargs):
            return websocket

        monkeypatch.setattr(module, "require_api_key", lambda config: "test-key")
        monkeypatch.setattr(module.websockets, "connect", connect)

        await model.connect(config.session)

        session = json.loads(websocket.sent[0])["session"]
        turn_detection = session["audio"]["input"]["turn_detection"]
        assert turn_detection == {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": silence_duration_ms,
            "create_response": True,
            "interrupt_response": True,
        }
        assert "gpt-realtime-2025-08-28" in model.endpoint(config.session)
        assert json.loads(websocket.sent[0])["session"]["type"] == "realtime"
        await model.close()

    asyncio.run(run())


def test_eot_session_disables_server_vad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gpt-realtime-fast"))
        model = load_adapter(config)
        websocket = EventWebSocket([{"type": "session.updated"}])
        module = sys.modules[model.__class__.__module__]

        async def connect(*args, **kwargs):
            return websocket

        monkeypatch.setattr(module, "require_api_key", lambda config: "test-key")
        monkeypatch.setattr(module.websockets, "connect", connect)

        session = SessionConfig(detect_turns=False)
        await model.connect(session)

        payload = json.loads(websocket.sent[0])["session"]
        assert payload["audio"]["input"]["turn_detection"] is None
        await model.close()

    asyncio.run(run())


def test_end_input_commits_buffer_and_creates_response() -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gpt-realtime-fast"))
        model = load_adapter(config)
        websocket = FakeWebSocket()
        model._ws = websocket

        await model.end_input()

        assert [json.loads(item)["type"] for item in websocket.sent] == [
            "input_audio_buffer.commit",
            "response.create",
        ]

    asyncio.run(run())


def test_missing_session_toml_keys_fail() -> None:
    config = load_model_config(resolve_model_ref("gpt-realtime-fast"))
    model = load_adapter(config)
    module = sys.modules[model.__class__.__module__]
    with pytest.raises(SystemExit, match="vad_threshold"):
        module.turn_detection(SessionConfig(detect_turns=True))
