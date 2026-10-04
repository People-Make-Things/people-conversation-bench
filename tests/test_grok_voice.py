"""Tests for Grok Voice (xAI) session layout on the shared realtime adapter."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from bench.protocol import SessionConfig
from bench.registry import load_adapter, load_model_config, resolve_model_ref


class EventWebSocket:
    def __init__(self, events: list[dict]) -> None:
        self.sent: list[str] = []
        self.events = iter(events)

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        try:
            return json.dumps(next(self.events))
        except StopIteration:
            raise StopAsyncIteration

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self) -> None:
        pass


def test_grok_voice_sends_xai_session_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("grok-voice"))
        model = load_adapter(config)
        websocket = EventWebSocket([{"type": "session.updated"}])
        module = sys.modules[model.__class__.__module__]
        requested: list[str] = []

        async def connect(*args, **kwargs):
            return websocket

        def require_api_key(config) -> str:
            requested.append(config.connection.api_key_env or "")
            return "test-key"

        monkeypatch.setattr(module, "require_api_key", require_api_key)
        monkeypatch.setattr(module.websockets, "connect", connect)

        await model.connect(config.session)

        assert config.session_layout == "xai"
        assert config.connection.api_key_env == "XAI_API_KEY"
        assert requested == ["XAI_API_KEY"]
        assert "api.x.ai" in model.endpoint(config.session)
        session = json.loads(websocket.sent[0])["session"]
        assert session["voice"] == "eve"
        assert session["turn_detection"] == {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 300,
        }
        assert "type" not in session
        assert "output_modalities" not in session
        assert session["audio"]["input"]["format"] == {
            "type": "audio/pcm",
            "rate": 24000,
        }
        await model.close()

    asyncio.run(run())


def test_grok_voice_eot_disables_server_vad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("grok-voice"))
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
        assert payload["turn_detection"] is None
        assert "type" not in payload
        await model.close()

    asyncio.run(run())


def test_session_layout_is_not_session_extra() -> None:
    config = load_model_config(resolve_model_ref("grok-voice"))
    model = load_adapter(config)
    module = sys.modules[model.__class__.__module__]
    session = SessionConfig(detect_turns=False)

    openai = module.openai_session_update(
        session, input_sample_rate=24000, output_sample_rate=24000
    )
    xai = module.xai_session_update(
        session, input_sample_rate=24000, output_sample_rate=24000
    )

    assert config.session_layout == "xai"
    assert "session_layout" not in config.session.extra
    assert "api_key_env" not in config.session.extra
    assert openai["session"]["type"] == "realtime"
    assert "type" not in xai["session"]
