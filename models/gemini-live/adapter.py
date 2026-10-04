"""Gemini Live WebSocket adapter."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from urllib.parse import urlencode, urlparse, urlunparse

import numpy as np
import websockets

from bench.audio import b64_to_pcm_float, pcm_float_to_b64
from bench.protocol import (
    AudioEvent,
    AudioFrame,
    AudioInterrupted,
    Context,
    ContextType,
    EventContext,
    ModelSignal,
    PreloadNotSupportedError,
    RealtimeModel,
    SessionConfig,
    SessionMetrics,
    SessionMode,
)
from bench.registry import ModelConfig, require_api_key, resolve_connection

LIVE_MODEL_PREFIX = "models/"
AUDIO_MIME = "audio/pcm;rate=24000"


def live_model_name(session: SessionConfig) -> str:
    name = session.extra_value("live_model").strip()
    if name.startswith(LIVE_MODEL_PREFIX):
        return name
    return f"{LIVE_MODEL_PREFIX}{name}"


def authenticated_url(base: str, api_key: str) -> str:
    parsed = urlparse(base)
    extra = urlencode({"key": api_key})
    query = f"{parsed.query}&{extra}" if parsed.query else extra
    return urlunparse(parsed._replace(query=query))


def context_turns(context: EventContext) -> list[dict]:
    turns: list[dict] = []
    for event in context.events:
        item = event["item"]
        role = "model" if item.get("role") == "assistant" else "user"
        parts: list[dict] = []
        for content in item.get("content", []):
            content_type = content.get("type")
            text = str(content.get("text") or content.get("transcript") or "").strip()
            if content_type in {"input_text", "output_text"} and text:
                parts.append({"text": text})
            elif content_type == "input_audio":
                if text:
                    parts.append({"text": text})
                audio = content.get("audio")
                if audio:
                    parts.append(
                        {
                            "inlineData": {
                                "mimeType": AUDIO_MIME,
                                "data": audio,
                            }
                        }
                    )
        if parts:
            turns.append({"role": role, "parts": parts})
    return turns


def setup_payload(session: SessionConfig, mode: SessionMode) -> dict:
    generation_config: dict = {"responseModalities": ["AUDIO"]}
    if session.voice_prompt.strip():
        generation_config["speechConfig"] = {
            "voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": session.voice_prompt.strip()}
            }
        }
    if session.detect_turns:
        interrupt = session.extra_value("interrupt_response").lower() == "true"
        realtime_input_config = {
            "automaticActivityDetection": {
                "disabled": False,
                "prefixPaddingMs": int(session.extra_value("vad_prefix_padding_ms")),
                "silenceDurationMs": int(session.extra_value("vad_silence_duration_ms")),
            },
            "activityHandling": (
                "START_OF_ACTIVITY_INTERRUPTS" if interrupt else "NO_INTERRUPTION"
            ),
        }
    else:
        realtime_input_config = {"automaticActivityDetection": {"disabled": True}}
    setup: dict = {
        "model": live_model_name(session),
        "generationConfig": generation_config,
        "outputAudioTranscription": {},
        "realtimeInputConfig": realtime_input_config,
    }
    instructions = session.text_prompt.strip()
    if instructions:
        setup["systemInstruction"] = {"parts": [{"text": instructions}]}
    if mode is SessionMode.PRELOAD:
        setup["historyConfig"] = {"initialHistoryInClientContent": True}
    return {"setup": setup}


class GeminiLiveModel(RealtimeModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self._metrics = SessionMetrics()
        self._ws: websockets.ClientConnection | None = None
        self._closed = False
        self._recv_task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue[AudioEvent | None] = asyncio.Queue()
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._signal_queue: asyncio.Queue[ModelSignal | None] = asyncio.Queue()
        self._session_ready = asyncio.Event()
        self._connect_error: str | None = None
        self._manual_turns = False
        self._activity_open = False

    @property
    def supported_contexts(self) -> tuple[ContextType, ...]:
        return (ContextType.TEXT_AUDIO, ContextType.TEXT)

    @property
    def metrics(self) -> SessionMetrics:
        return self._metrics

    def endpoint(self, session: SessionConfig) -> str:
        return resolve_connection(self.config)

    async def connect(
        self,
        session: SessionConfig,
        *,
        mode: SessionMode = SessionMode.LIVE,
    ) -> None:
        api_key = require_api_key(self.config)
        url = self.endpoint(session)
        print(f"Connecting to {url}")
        self._manual_turns = not session.detect_turns
        self._ws = await websockets.connect(
            authenticated_url(url, api_key),
            max_size=None,
            open_timeout=60,
            close_timeout=1,
        )
        self._recv_task = asyncio.create_task(self.recv_loop())
        await self._ws.send(json.dumps(setup_payload(session, mode)))
        await asyncio.wait_for(self._session_ready.wait(), timeout=10)
        if self._connect_error:
            raise RuntimeError(self._connect_error)

    async def send_audio(self, pcm: np.ndarray) -> None:
        if self._closed or self._ws is None or pcm.size == 0:
            return
        if self._metrics.first_audio_sent_at is None:
            self._metrics.first_audio_sent_at = time.monotonic()
        if self._manual_turns and not self._activity_open:
            await self._ws.send(
                json.dumps({"realtimeInput": {"activityStart": {}}})
            )
            self._activity_open = True
        await self._ws.send(
            json.dumps(
                {
                    "realtimeInput": {
                        "audio": {
                            "mimeType": AUDIO_MIME,
                            "data": pcm_float_to_b64(pcm),
                        }
                    }
                }
            )
        )

    async def end_input(self) -> None:
        if self._closed or self._ws is None:
            return
        if self._manual_turns and self._activity_open:
            await self._ws.send(
                json.dumps({"realtimeInput": {"activityEnd": {}}})
            )
            self._activity_open = False

    async def _put_audio(self, pcm: np.ndarray) -> None:
        if pcm.size == 0:
            return
        if self._metrics.first_audio_received_at is None:
            self._metrics.first_audio_received_at = time.monotonic()
            if self._metrics.first_audio_sent_at is not None:
                delta_s = (
                    self._metrics.first_audio_received_at
                    - self._metrics.first_audio_sent_at
                )
                self._metrics.ttfb_ms = delta_s * 1000
        await self._audio_queue.put(AudioFrame(pcm=pcm))

    async def dispatch_event(self, event: dict) -> None:
        error = event.get("error")
        if error:
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            print(f"[WARN] gemini live error: {message}", flush=True)
            if not self._session_ready.is_set():
                self._connect_error = message
                self._session_ready.set()
            return

        if event.get("setupComplete") is not None:
            self._session_ready.set()
            return

        content = event.get("serverContent")
        if not isinstance(content, dict):
            return

        if content.get("interrupted"):
            print("[INTERRUPT] serverContent.interrupted", flush=True)
            await self._audio_queue.put(AudioInterrupted())
            await self._signal_queue.put(
                ModelSignal(kind="interrupted", received_at=time.monotonic())
            )

        turn = content.get("modelTurn") or {}
        for part in turn.get("parts") or []:
            inline = part.get("inlineData") or {}
            data = inline.get("data")
            if data:
                await self._put_audio(b64_to_pcm_float(data))
            text = part.get("text")
            if text:
                await self._text_queue.put(text)

        transcription = content.get("outputTranscription")
        if isinstance(transcription, dict):
            text = transcription.get("text")
            if text:
                await self._text_queue.put(text)

        if content.get("generationComplete"):
            await self._signal_queue.put(
                ModelSignal(
                    kind="response_done",
                    received_at=time.monotonic(),
                    data={"status": "generation_complete"},
                )
            )
        if content.get("turnComplete"):
            await self._signal_queue.put(
                ModelSignal(kind="turn_complete", received_at=time.monotonic())
            )

    async def recv_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            if self._closed:
                continue
            payload = raw if isinstance(raw, str) else raw.decode()
            await self.dispatch_event(json.loads(payload))

        await self._audio_queue.put(None)
        await self._text_queue.put(None)
        await self._signal_queue.put(None)

    async def recv_audio(self) -> AsyncIterator[AudioEvent]:
        while True:
            frame = await self._audio_queue.get()
            if frame is None:
                break
            yield frame

    async def recv_text(self) -> AsyncIterator[str]:
        while True:
            token = await self._text_queue.get()
            if token is None:
                break
            yield token

    async def recv_signals(self) -> AsyncIterator[ModelSignal]:
        while True:
            signal = await self._signal_queue.get()
            if signal is None:
                break
            yield signal

    async def preload_context(self, context: Context) -> None:
        if not isinstance(context, EventContext):
            raise PreloadNotSupportedError(
                f"GeminiLiveModel does not support {context.kind.value} context preloading"
            )
        if self._ws is None:
            raise RuntimeError("connect() must be called before preload_context()")
        turns = context_turns(context)
        print(
            f"Preloading {context.kind.value} context ({len(turns)} turns)",
            flush=True,
        )
        await self._ws.send(
            json.dumps(
                {
                    "clientContent": {
                        "turns": turns,
                        "turnComplete": True,
                    }
                }
            )
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._recv_task is not None:
            self._recv_task.cancel()
        if self._ws is not None:
            await self._ws.close()
        await self._audio_queue.put(None)
        await self._text_queue.put(None)
        await self._signal_queue.put(None)


def create(config: ModelConfig) -> GeminiLiveModel:
    return GeminiLiveModel(config)
