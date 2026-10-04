"""Client-side adapter behaviour that the loopback tests cannot reach.

These cover the failure modes that need a hostile peer: a server that talks
before it acknowledges the prefill, a socket that dies mid-stream, and an sphn
whose method names moved.
"""

from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest
import websockets

from bench.audio import MODEL_FRAME_SAMPLES, pcm16_bytes_to_float, pcm_float_to_int16_bytes
from bench.protocol import SessionConfig, SessionMode
from bench.registry import AudioConfig, ConnectionConfig, ModelConfig
from models.personaplex.adapter import (
    PersonaPlexModel,
    build_ws_url,
    retryable_connect_error,
)
from models.personaplex.opus_codec import bind_missing_method
from models.personaplex.server.wire import MSG_PRIMED, MSG_TEXT


def config() -> ModelConfig:
    return ModelConfig(
        id="personaplex",
        name="PersonaPlex",
        model_dir=None,
        adapter_path=None,
        connection=ConnectionConfig(url="http://localhost:1"),
        audio=AudioConfig(
            input_sample_rate=24000,
            output_sample_rate=24000,
            encoding="opus",
            frame_samples=MODEL_FRAME_SAMPLES,
        ),
        session=SessionConfig(),
    )


class ChattyServer:
    """Sends nothing until asked, then acknowledges the prefill."""

    def __init__(self) -> None:
        self.recv_calls = 0

    async def recv(self) -> bytes:
        self.recv_calls += 1
        return bytes([MSG_PRIMED]) + json.dumps({"history_duration_sec": 1.0}).encode()


def test_wait_for_primed_does_not_spin_on_a_message_it_cannot_use() -> None:
    """The skipped message used to be pushed back onto the queue being drained,
    so a single early text token span the loop at 100% CPU forever."""
    model = PersonaPlexModel(config())
    early = bytes([MSG_TEXT]) + b"hello"
    model._early_messages = [early]
    server = ChattyServer()

    asyncio.run(asyncio.wait_for(model._wait_for_primed(server), timeout=5))

    assert server.recv_calls == 1
    assert model._early_messages == [early]


def test_bind_missing_method_raises_instead_of_binding_silence() -> None:
    """A no-op fallback on the only audio path is a rollout of pure silence
    that still scores, with nothing anywhere to say it went wrong."""

    class Renamed:
        def get_bytes(self) -> bytes:
            return b"ok"

    class Missing:
        pass

    assert bind_missing_method(Renamed(), "read_bytes", ("get_bytes",))() == b"ok"
    with pytest.raises(AttributeError, match="unsupported"):
        bind_missing_method(Missing(), "read_bytes", ("get_bytes",))


def test_trailing_model_audio_is_flushed_on_close() -> None:
    """The accumulator holds a sub-frame remainder, which at end of stream is
    the model's last word rather than a partial frame to wait on."""
    model = PersonaPlexModel(config())
    model._frames.push(np.full(MODEL_FRAME_SAMPLES // 2, 0.5, dtype=np.float32))

    async def run() -> np.ndarray:
        await model._flush_opus_frames()
        return (await model._audio_queue.get()).pcm

    flushed = asyncio.run(run())
    assert flushed.shape[0] == MODEL_FRAME_SAMPLES
    assert np.count_nonzero(flushed) == MODEL_FRAME_SAMPLES // 2


def test_pcm16_round_trip_is_symmetric() -> None:
    """Encode scaled by 32767 while decode divided by 32768, and the cast
    truncated toward zero, so every round trip lost up to a full step."""
    pcm = np.linspace(-1.0, 1.0, 4001, dtype=np.float32)
    restored = pcm16_bytes_to_float(pcm_float_to_int16_bytes(pcm))

    assert np.max(np.abs(restored - pcm)) <= 1.0 / 32768.0


def test_preload_url_drops_the_query_string() -> None:
    session = SessionConfig(voice_prompt="NATF2.pt", text_prompt="p" * 9000, seed=3)
    url = build_ws_url("http://host", session, path="/api/duplex", include_session=False)

    assert url == "ws://host/api/duplex"
    assert len(url) < 8190


def test_connection_error_is_latched_for_the_rollout_to_see() -> None:
    """send_audio only fills the Opus writer, so a dead socket is otherwise
    invisible and the rollout records input frames that never left."""
    model = PersonaPlexModel(config())
    assert model.connection_error is None

    error = websockets.ConnectionClosedError(None, None)
    model._io_error = error
    assert model.connection_error is error


def test_retryable_connect_error_is_only_the_cold_proxy_cases() -> None:
    assert retryable_connect_error(TimeoutError("timed out during opening handshake"))
    assert retryable_connect_error(ConnectionResetError(54, "Connection reset by peer"))
    assert retryable_connect_error(websockets.ConnectionClosedOK(None, None))
    assert not retryable_connect_error(RuntimeError("handshake protocol broke"))


def test_connect_retries_a_stuck_handshake_then_succeeds(monkeypatch) -> None:
    calls = {"n": 0}

    class FakeSocket:
        pass

    async def fake_connect(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] < 3:
            raise TimeoutError("timed out during opening handshake")
        return FakeSocket()

    monkeypatch.setattr("models.personaplex.adapter.websockets.connect", fake_connect)
    model = PersonaPlexModel(config())

    asyncio.run(model.connect(SessionConfig(), mode=SessionMode.PRELOAD))

    assert calls["n"] == 3
    assert isinstance(model._ws, FakeSocket)
