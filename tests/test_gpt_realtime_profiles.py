"""Tests for GPT Realtime VAD profiles."""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from bench.registry import load_adapter, load_model_config, resolve_model_ref


class FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.events = iter(({"type": "session.updated"},))

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
        websocket = FakeWebSocket()
        module = sys.modules[model.__class__.__module__]

        async def connect(*args, **kwargs):
            return websocket

        monkeypatch.setattr(module, "require_api_key", lambda config: "test-key")
        monkeypatch.setattr(module.websockets, "connect", connect)

        await model.connect(config.session)

        session = json.loads(websocket.sent[0])["session"]
        assert session["audio"]["input"]["turn_detection"] == {
            "type": "server_vad",
            "threshold": 0.5,
            "prefix_padding_ms": 300,
            "silence_duration_ms": silence_duration_ms,
            "create_response": True,
            "interrupt_response": True,
        }
        await model.close()

    asyncio.run(run())


def test_semantic_vad_profile_sends_low_eagerness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run() -> None:
        config = load_model_config(resolve_model_ref("gpt-realtime-semantic-low"))
        model = load_adapter(config)
        websocket = FakeWebSocket()
        module = sys.modules[model.__class__.__module__]

        async def connect(*args, **kwargs):
            return websocket

        monkeypatch.setattr(module, "require_api_key", lambda config: "test-key")
        monkeypatch.setattr(module.websockets, "connect", connect)

        await model.connect(config.session)

        session = json.loads(websocket.sent[0])["session"]
        assert session["audio"]["input"]["turn_detection"] == {
            "type": "semantic_vad",
            "eagerness": "low",
            "create_response": True,
            "interrupt_response": True,
        }
        await model.close()

    asyncio.run(run())
