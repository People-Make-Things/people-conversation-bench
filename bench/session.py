"""Shared realtime model lifecycle for interactive and headless sessions."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

import numpy as np

from bench.protocol import (
    AudioEvent,
    Context,
    ModelSignal,
    RealtimeModel,
    SessionConfig,
    SessionMode,
)

AudioHandler = Callable[[AudioEvent, float], Awaitable[None]]
TextHandler = Callable[[str, float], Awaitable[None]]
SignalHandler = Callable[[ModelSignal], Awaitable[None]]


async def ignore_audio(event: AudioEvent, received_at: float) -> None:
    pass


async def ignore_text(text: str, received_at: float) -> None:
    pass


async def ignore_signal(signal: ModelSignal) -> None:
    pass


class RealtimeSession:
    def __init__(
        self,
        model: RealtimeModel,
        config: SessionConfig,
        *,
        on_audio: AudioHandler = ignore_audio,
        on_text: TextHandler = ignore_text,
        on_signal: SignalHandler = ignore_signal,
    ):
        self.model = model
        self.config = config
        self.on_audio = on_audio
        self.on_text = on_text
        self.on_signal = on_signal
        self.tasks: list[asyncio.Task] = []

    async def start(self, context: Context | None = None) -> None:
        mode = SessionMode.PRELOAD if context is not None else SessionMode.LIVE
        await self.model.connect(self.config, mode=mode)
        if context is not None:
            await self.model.preload_context(context)
        self.tasks = [
            asyncio.create_task(self.receive_audio()),
            asyncio.create_task(self.receive_text()),
            asyncio.create_task(self.receive_signals()),
        ]

    async def send_audio(self, pcm: np.ndarray) -> None:
        await self.model.send_audio(pcm)

    async def end_input(self) -> None:
        await self.model.end_input()

    async def receive_audio(self) -> None:
        async for event in self.model.recv_audio():
            await self.on_audio(event, time.monotonic())

    async def receive_text(self) -> None:
        async for text in self.model.recv_text():
            await self.on_text(text, time.monotonic())

    async def receive_signals(self) -> None:
        async for signal in self.model.recv_signals():
            await self.on_signal(signal)

    async def close(self) -> None:
        # One loop turn so the receive tasks trace events the model already
        # delivered: the pause drain returns right after its last frame, and
        # cancelling first races against a frame one await from being traced.
        # Covers the single-await case only; a deeper drain would change close
        # timing.
        await asyncio.sleep(0)
        for task in self.tasks:
            task.cancel()
        await self.model.close()
        await asyncio.gather(*self.tasks, return_exceptions=True)
