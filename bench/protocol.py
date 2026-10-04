"""Shared interface for realtime voice model adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator

import numpy as np


class PreloadNotSupportedError(ValueError):
    """Raised when a model adapter cannot preload the requested context type."""


class SessionMode(Enum):
    LIVE = "live"
    PRELOAD = "preload"


class ContextType(Enum):
    TEXT = "text"
    TEXT_AUDIO = "text_audio"
    DUPLEX_AUDIO = "duplex_audio"


@dataclass(frozen=True)
class TimedWord:
    text: str
    start_sec: float
    end_sec: float | None = None

    def to_dict(self) -> dict:
        return {"text": self.text, "start_sec": self.start_sec, "end_sec": self.end_sec}

    @classmethod
    def from_dict(cls, entry: dict) -> TimedWord:
        return cls(
            text=str(entry.get("text", "")).strip(),
            start_sec=float(entry["start_sec"]),
            end_sec=(
                float(entry["end_sec"]) if entry.get("end_sec") is not None else None
            ),
        )


def classify_event_context_kind(events: tuple[dict, ...]) -> ContextType:
    for event in events:
        for content in event.get("item", {}).get("content", []):
            if content.get("type") == "input_audio":
                return ContextType.TEXT_AUDIO
    return ContextType.TEXT


@dataclass(frozen=True)
class EventContext:
    events: tuple[dict, ...]
    kind: ContextType


def event_context(events: tuple[dict, ...] | list[dict]) -> EventContext:
    normalized = tuple(events)
    kind = classify_event_context_kind(normalized)
    if kind not in (ContextType.TEXT, ContextType.TEXT_AUDIO):
        raise ValueError(f"unexpected event context kind: {kind}")
    return EventContext(events=normalized, kind=kind)


@dataclass(frozen=True)
class DuplexAudioContext:
    user_pcm: np.ndarray
    assistant_pcm: np.ndarray
    sample_rate: int
    words: tuple[TimedWord, ...] = ()

    @property
    def kind(self) -> ContextType:
        return ContextType.DUPLEX_AUDIO


Context = EventContext | DuplexAudioContext


@dataclass
class SessionConfig:
    voice_prompt: str = ""
    text_prompt: str = ""
    seed: int | None = None
    extra: dict[str, str] = field(default_factory=dict)
    detect_turns: bool = True

    def extra_value(self, name: str) -> str:
        value = self.extra.get(name)
        if value is None or value == "":
            raise SystemExit(f"Missing [session] {name} in model.toml")
        return value


@dataclass
class SessionMetrics:
    ttfb_ms: float | None = None
    first_audio_sent_at: float | None = None
    first_audio_received_at: float | None = None


@dataclass
class AudioFrame:
    pcm: np.ndarray
    on_played: Callable[[int], None] | None = field(default=None, repr=False)

    def played(self, samples: int) -> None:
        if self.on_played is not None:
            self.on_played(samples)


@dataclass(frozen=True)
class AudioInterrupted:
    pass


AudioEvent = AudioFrame | AudioInterrupted


@dataclass(frozen=True)
class ModelSignal:
    kind: str
    received_at: float
    data: dict = field(default_factory=dict)


async def empty_model_signals() -> AsyncIterator[ModelSignal]:
    if False:
        yield ModelSignal("", 0.0)


class RealtimeModel(ABC):
    @property
    @abstractmethod
    def supported_contexts(self) -> tuple[ContextType, ...]: ...

    @abstractmethod
    def endpoint(self, session: SessionConfig) -> str: ...

    @abstractmethod
    async def connect(
        self,
        session: SessionConfig,
        *,
        mode: SessionMode = SessionMode.LIVE,
    ) -> None: ...

    @abstractmethod
    async def send_audio(self, pcm: np.ndarray) -> None: ...

    async def end_input(self) -> None:
        return

    @abstractmethod
    def recv_audio(self) -> AsyncIterator[AudioEvent]: ...

    @abstractmethod
    def recv_text(self) -> AsyncIterator[str]: ...

    def recv_signals(self) -> AsyncIterator[ModelSignal]:
        return empty_model_signals()

    @abstractmethod
    async def close(self) -> None: ...

    @property
    def connection_error(self) -> BaseException | None:
        """Exception that terminated the transport, if it died on its own."""
        return None

    @property
    @abstractmethod
    def metrics(self) -> SessionMetrics: ...

    async def preload_context(self, context: Context) -> None:
        raise PreloadNotSupportedError(
            f"{type(self).__name__} does not support {context.kind.value} context preloading"
        )
