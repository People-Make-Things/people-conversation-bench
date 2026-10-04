from __future__ import annotations

import asyncio
from pathlib import Path

from bench.protocol import ContextType, SessionConfig
from bench.registry import AudioConfig, ConnectionConfig, ModelConfig
from bench.warmup import (
    duplex_socket_url,
    raise_open_files,
    warmup_duplex,
    warmup_model,
)


def test_duplex_socket_url_uses_wss_and_duplex_path() -> None:
    assert (
        duplex_socket_url("https://example.modal.run")
        == "wss://example.modal.run/api/duplex"
    )


def test_raise_open_files_does_not_raise() -> None:
    raise_open_files()


def test_warmup_skips_hosted_models() -> None:
    config = ModelConfig(
        id="gpt-realtime-fast",
        name="GPT",
        model_dir=Path("."),
        adapter_path=Path("adapter.py"),
        connection=ConnectionConfig(url="wss://example"),
        audio=AudioConfig(
            input_sample_rate=24000,
            output_sample_rate=24000,
            encoding="pcm16",
            frame_samples=480,
        ),
        session=SessionConfig(),
        max_concurrency=8,
    )

    assert asyncio.run(warmup_model(config, ())) == 0
    assert asyncio.run(warmup_model(config, (ContextType.TEXT,))) == 0


def test_warmup_holds_all_sockets_then_closes(monkeypatch) -> None:
    held = 0
    peak = 0
    releases = 0

    class FakeSocket:
        async def __aenter__(self):
            nonlocal held, peak
            held += 1
            peak = max(peak, held)
            return self

        async def __aexit__(self, *_args):
            nonlocal held, releases
            held -= 1
            releases += 1
            return False

    def fake_connect(*_args, **_kwargs):
        return FakeSocket()

    monkeypatch.setattr("bench.warmup.websockets.connect", fake_connect)

    warmed = asyncio.run(warmup_duplex("wss://example/api/duplex", 4, "personaplex"))

    assert warmed == 4
    assert peak == 4
    assert held == 0
    assert releases == 4
