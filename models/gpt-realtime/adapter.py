"""OpenAI Realtime WebSocket adapter."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

import numpy as np
import websockets

from bench.audio import b64_to_pcm_float, pcm_float_to_b64
from bench.protocol import (
    AudioEvent,
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

from .playback import PlaybackTracker


def turn_detection(session: SessionConfig) -> dict | None:
    if not session.detect_turns:
        return None
    vad_type = session.extra.get("vad_type") or "server_vad"
    if vad_type == "semantic_vad":
        return {
            "type": "semantic_vad",
            "eagerness": session.extra_value("vad_eagerness"),
        }
    if vad_type != "server_vad":
        raise SystemExit(f"Unknown [session] vad_type: {vad_type}")
    return {
        "type": "server_vad",
        "threshold": float(session.extra_value("vad_threshold")),
        "prefix_padding_ms": int(session.extra_value("vad_prefix_padding_ms")),
        "silence_duration_ms": int(session.extra_value("vad_silence_duration_ms")),
    }


def attach_instructions(payload: dict, session: SessionConfig) -> dict:
    instructions = session.text_prompt.strip()
    if instructions:
        payload["session"]["instructions"] = instructions
    return payload


def openai_session_update(
    session: SessionConfig,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
) -> dict:
    detection = turn_detection(session)
    if detection is not None:
        detection = {
            **detection,
            "create_response": True,
            "interrupt_response": (
                session.extra_value("interrupt_response").lower() == "true"
            ),
        }
    return attach_instructions(
        {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": input_sample_rate},
                        "turn_detection": detection,
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": output_sample_rate},
                        "voice": session.voice_prompt,
                    },
                },
            },
        },
        session,
    )


def xai_session_update(
    session: SessionConfig,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
) -> dict:
    return attach_instructions(
        {
            "type": "session.update",
            "session": {
                "voice": session.voice_prompt,
                "turn_detection": turn_detection(session),
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": input_sample_rate}
                    },
                    "output": {
                        "format": {"type": "audio/pcm", "rate": output_sample_rate}
                    },
                },
            },
        },
        session,
    )


def session_update_payload(
    session: SessionConfig,
    *,
    input_sample_rate: int,
    output_sample_rate: int,
    layout: str,
) -> dict:
    if layout == "xai":
        return xai_session_update(
            session,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
        )
    if layout == "openai":
        return openai_session_update(
            session,
            input_sample_rate=input_sample_rate,
            output_sample_rate=output_sample_rate,
        )
    raise ValueError(f"unknown session_layout: {layout}")


class GPTRealtimeModel(RealtimeModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self._metrics = SessionMetrics()
        self._ws: websockets.ClientConnection | None = None
        self._closed = False
        self._recv_task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue[AudioEvent | None] = asyncio.Queue()
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._signal_queue: asyncio.Queue[ModelSignal | None] = asyncio.Queue()
        self._frame_samples = config.audio.frame_samples
        self._session_ready = asyncio.Event()
        self._playback = PlaybackTracker(
            config.audio.output_sample_rate,
            config.audio.frame_samples,
        )
        self._interrupt_response = True

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
        del mode
        api_key = require_api_key(self.config)
        url = self.endpoint(session)

        print(f"Connecting to {url}")
        self._ws = await websockets.connect(
            url,
            additional_headers={
                "Authorization": f"Bearer {api_key}",
            },
            max_size=None,
            open_timeout=60,
            close_timeout=1,
        )

        self._interrupt_response = False
        if session.detect_turns:
            self._interrupt_response = (
                session.extra_value("interrupt_response").lower() == "true"
            )
        payload = session_update_payload(
            session,
            input_sample_rate=self.config.audio.input_sample_rate,
            output_sample_rate=self.config.audio.output_sample_rate,
            layout=self.config.session_layout,
        )

        self._recv_task = asyncio.create_task(self.recv_loop())
        await self._ws.send(json.dumps(payload))
        await asyncio.wait_for(self._session_ready.wait(), timeout=10)

    async def send_audio(self, pcm: np.ndarray) -> None:
        if self._closed or self._ws is None or pcm.size == 0:
            return
        if self._metrics.first_audio_sent_at is None:
            self._metrics.first_audio_sent_at = time.monotonic()
        await self._ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "audio": pcm_float_to_b64(pcm),
                }
            )
        )

    async def end_input(self) -> None:
        if self._closed or self._ws is None:
            return
        await self._ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await self._ws.send(json.dumps({"type": "response.create"}))

    async def dispatch_event(self, event: dict) -> None:
        event_type = event.get("type")

        if event_type == "session.updated":
            self._session_ready.set()
            return

        if event_type == "input_audio_buffer.speech_started":
            await self._signal_queue.put(
                ModelSignal(
                    kind="input_speech_started",
                    received_at=time.monotonic(),
                    data={
                        "item_id": event.get("item_id"),
                        "audio_start_ms": event.get("audio_start_ms"),
                    },
                )
            )
            print(
                "[VAD] input_audio_buffer.speech_started "
                f"(item_id={event.get('item_id')}, "
                f"audio_start_ms={event.get('audio_start_ms')})",
                flush=True,
            )
            await self._playback.handle_interruption(
                self._ws,
                self._audio_queue,
                self._interrupt_response,
            )
            return

        if event_type == "input_audio_buffer.speech_stopped":
            await self._signal_queue.put(
                ModelSignal(
                    kind="input_speech_stopped",
                    received_at=time.monotonic(),
                    data={
                        "item_id": event.get("item_id"),
                        "audio_end_ms": event.get("audio_end_ms"),
                    },
                )
            )
            print(
                "[VAD] input_audio_buffer.speech_stopped "
                f"(item_id={event.get('item_id')}, "
                f"audio_end_ms={event.get('audio_end_ms')})",
                flush=True,
            )
            return

        if event_type == "response.created":
            response_id = event.get("response", {}).get("id")
            self._playback.track_response_created(response_id)
            await self._signal_queue.put(
                ModelSignal(
                    kind="response_created",
                    received_at=time.monotonic(),
                    data={"response_id": response_id},
                )
            )
            return

        if event_type == "response.done":
            response_id = event.get("response", {}).get("id")
            status = event.get("response", {}).get("status")
            if status and status != "completed":
                print(
                    f"[INTERRUPT] response.done status={status} "
                    f"(response_id={response_id})",
                    flush=True,
                )
            self._playback.track_response_done(response_id)
            await self._signal_queue.put(
                ModelSignal(
                    kind="response_done",
                    received_at=time.monotonic(),
                    data={"response_id": response_id, "status": status},
                )
            )
            return

        if event_type in {"response.output_audio.delta", "response.audio.delta"}:
            response_id = event.get("response_id")
            item_id = event.get("item_id")
            if not self._playback.should_accept_response(response_id) or not item_id:
                return
            delta = event.get("delta", "")
            if not delta:
                return
            if self._metrics.first_audio_received_at is None:
                self._metrics.first_audio_received_at = time.monotonic()
                if self._metrics.first_audio_sent_at is not None:
                    delta_s = (
                        self._metrics.first_audio_received_at
                        - self._metrics.first_audio_sent_at
                    )
                    self._metrics.ttfb_ms = delta_s * 1000
            await self._playback.push_audio_delta(
                b64_to_pcm_float(delta),
                response_id,
                item_id,
                event.get("content_index", 0),
                self._audio_queue,
            )
            return

        if event_type in {
            "response.output_text.delta",
            "response.output_audio_transcript.delta",
            "response.text.delta",
            "response.audio_transcript.delta",
        }:
            response_id = event.get("response_id")
            if not self._playback.should_accept_response(response_id):
                return
            delta = event.get("delta", "")
            if delta:
                await self._text_queue.put(delta)
            return

        if event_type == "error":
            error = event.get("error", {})
            message = error.get("message", str(error))
            print(f"[WARN] realtime error: {message}", flush=True)

    async def recv_loop(self) -> None:
        assert self._ws is not None
        async for raw in self._ws:
            if self._closed or not isinstance(raw, str):
                continue
            await self.dispatch_event(json.loads(raw))

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
                f"GPTRealtimeModel does not support {context.kind.value} context preloading"
            )
        if self._ws is None:
            raise RuntimeError("connect() must be called before preload_context()")
        print(
            f"Preloading {context.kind.value} context ({len(context.events)} items)",
            flush=True,
        )
        for event in context.events:
            await self._ws.send(json.dumps(event))

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


def create(config: ModelConfig) -> GPTRealtimeModel:
    return GPTRealtimeModel(config)
