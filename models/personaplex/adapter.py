"""PersonaPlex WebSocket adapter (moshi.server wire protocol)."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from urllib.parse import urlencode, urlparse, urlunparse

import numpy as np
import websockets
from tqdm import tqdm

from bench.audio import FrameAccumulator, pcm_float_to_int16_bytes
from bench.protocol import (
    AudioEvent,
    AudioFrame,
    Context,
    ContextType,
    DuplexAudioContext,
    ModelSignal,
    PreloadNotSupportedError,
    RealtimeModel,
    SessionConfig,
    SessionMetrics,
    SessionMode,
)
from bench.registry import ModelConfig, resolve_connection

from .opus_codec import OpusStreams
from .server.wire import (
    MSG_ASSISTANT_HISTORY,
    MSG_COMMIT,
    MSG_META,
    MSG_PRIMED,
    MSG_TRANSCRIPT,
    MSG_USER_HISTORY,
    build_meta_payload,
    build_transcript_payload,
    iter_byte_chunks,
)


def build_ws_url(
    base_url: str,
    session: SessionConfig,
    path: str = "/api/chat",
    include_session: bool = True,
) -> str:
    if "://" in base_url:
        proto, rest = base_url.split("://", 1)
        ws_proto = "ws" if proto in {"http", "ws"} else "wss"
        base = f"{ws_proto}://{rest.rstrip('/')}"
    elif ":" not in base_url:
        base = f"ws://{base_url}:8998"
    else:
        base = f"ws://{base_url}"

    parsed = urlparse(base)
    if not include_session:
        # A persona text prompt can exceed the 8190-byte request line aiohttp
        # accepts; the duplex handshake carries the same fields in MSG_META.
        return urlunparse(parsed._replace(path=path, query=""))
    params = {
        "voice_prompt": session.voice_prompt,
        "text_prompt": session.text_prompt,
    }
    if session.seed is not None:
        params["seed"] = str(session.seed)

    query = urlencode(params)
    return urlunparse(parsed._replace(path=path, query=query))


# A handshake queued on a cold Modal proxy is never handed to a container that
# comes up later. A new connect is. Wave in-flight opens so 64 rollouts do not
# recreate that herd.
CONNECT_ATTEMPTS = 5
CONNECT_OPEN_TIMEOUT_SEC = 90.0
CONNECT_RETRY_SLEEP_SEC = 1.0
CONNECT_WAVE_SIZE = 8
_connect_gates: dict[str, asyncio.Semaphore] = {}


def handshake_gate(url: str) -> asyncio.Semaphore:
    key = urlparse(url).netloc
    gate = _connect_gates.get(key)
    if gate is None:
        gate = asyncio.Semaphore(CONNECT_WAVE_SIZE)
        _connect_gates[key] = gate
    return gate


def retryable_connect_error(error: BaseException) -> bool:
    if isinstance(
        error, (TimeoutError, ConnectionResetError, ConnectionError, OSError)
    ):
        return True
    if isinstance(error, websockets.ConnectionClosed):
        return True
    if type(error).__name__ == "InvalidMessage":
        return True
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status is not None and status >= 500


class PersonaPlexModel(RealtimeModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self._metrics = SessionMetrics()
        self._ws: websockets.ClientConnection | None = None
        self._closed = False
        self._mode = SessionMode.LIVE
        self._send_task: asyncio.Task | None = None
        self._recv_task: asyncio.Task | None = None
        self._audio_queue: asyncio.Queue[AudioEvent | None] = asyncio.Queue()
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._opus = OpusStreams(
            config.audio.input_sample_rate,
            config.audio.output_sample_rate,
        )
        self._frames = FrameAccumulator(config.audio.frame_samples)
        self._session: SessionConfig | None = None
        self._early_messages: list[bytes | str] = []
        self._signal_queue: asyncio.Queue[ModelSignal | None] = asyncio.Queue()
        self._io_error: BaseException | None = None

    @property
    def supported_contexts(self) -> tuple[ContextType, ...]:
        return (ContextType.DUPLEX_AUDIO,)

    @property
    def metrics(self) -> SessionMetrics:
        return self._metrics

    @property
    def connection_error(self) -> BaseException | None:
        return self._io_error

    def endpoint(self, session: SessionConfig) -> str:
        return build_ws_url(resolve_connection(self.config), session)

    async def connect(
        self,
        session: SessionConfig,
        *,
        mode: SessionMode = SessionMode.LIVE,
    ) -> None:
        self._session = session
        self._mode = mode
        preload = mode is SessionMode.PRELOAD
        url = build_ws_url(
            resolve_connection(self.config),
            session,
            path="/api/duplex" if preload else "/api/chat",
            include_session=not preload,
        )
        print(f"Connecting to {url}")
        print("Cold start can take a few minutes on first request...", flush=True)

        self._ws = await self._open_socket(url)
        if mode is SessionMode.PRELOAD:
            return

        await self._wait_for_handshake(self._ws)
        self._start_io_tasks()

    async def _open_socket(self, url: str):
        last_error: BaseException | None = None
        for attempt in range(1, CONNECT_ATTEMPTS + 1):
            try:
                async with handshake_gate(url):
                    return await websockets.connect(
                        url,
                        max_size=None,
                        open_timeout=CONNECT_OPEN_TIMEOUT_SEC,
                        close_timeout=1,
                        # Prefill runs a full LM step per history frame, so the
                        # server can legitimately go quiet for tens of seconds.
                        # Keepalive pings here only kill working sessions.
                        ping_interval=None,
                    )
            except Exception as error:
                last_error = error
                if not retryable_connect_error(error) or attempt == CONNECT_ATTEMPTS:
                    raise
                print(
                    f"connect attempt {attempt}/{CONNECT_ATTEMPTS} failed: "
                    f"{error!r}; retrying",
                    flush=True,
                )
                await asyncio.sleep(CONNECT_RETRY_SLEEP_SEC)
        assert last_error is not None
        raise last_error

    def _start_io_tasks(self) -> None:
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def preload_context(self, context: Context) -> None:
        if not isinstance(context, DuplexAudioContext):
            raise PreloadNotSupportedError(
                f"PersonaPlexModel does not support {context.kind.value} context preloading"
            )
        if self._mode is not SessionMode.PRELOAD or self._ws is None:
            raise RuntimeError("duplex preload requires connect(..., mode=SessionMode.PRELOAD)")
        if self._session is None:
            raise RuntimeError("connect() must be called before preload_context()")

        print(
            "Preloading duplex audio context "
            f"({context.user_pcm.shape[0] / context.sample_rate:.2f}s)",
            flush=True,
        )
        session = self._session
        await self._ws.send(
            bytes([MSG_META])
            + build_meta_payload(session.voice_prompt, session.text_prompt, session.seed)
        )
        user_payload = pcm_float_to_int16_bytes(context.user_pcm)
        assistant_payload = pcm_float_to_int16_bytes(context.assistant_pcm)
        upload_total = len(user_payload) + len(assistant_payload)
        with tqdm(
            total=upload_total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="Upload history",
            leave=True,
        ) as upload_bar:
            await self._send_pcm_payload(MSG_USER_HISTORY, user_payload, upload_bar)
            await self._send_pcm_payload(MSG_ASSISTANT_HISTORY, assistant_payload, upload_bar)
        await self._ws.send(
            bytes([MSG_TRANSCRIPT]) + build_transcript_payload(context.words)
        )
        await self._ws.send(bytes([MSG_COMMIT]))
        await self._wait_for_primed(
            self._ws,
            history_duration_sec=context.user_pcm.shape[0] / context.sample_rate,
        )
        self._start_io_tasks()

    async def _send_pcm_payload(
        self,
        msg_type: int,
        payload: bytes,
        progress: tqdm,
    ) -> None:
        assert self._ws is not None
        for chunk in iter_byte_chunks(payload):
            await self._ws.send(bytes([msg_type]) + chunk)
            progress.update(len(chunk))

    async def _recv_next(self, ws: websockets.ClientConnection) -> bytes | str:
        if self._early_messages:
            return self._early_messages.pop(0)
        return await ws.recv()

    async def _wait_for_handshake(self, ws: websockets.ClientConnection) -> None:
        print("Waiting for server handshake (model may still be loading)...", flush=True)
        while not self._closed:
            try:
                first = await asyncio.wait_for(self._recv_next(ws), timeout=300)
            except TimeoutError as exc:
                raise RuntimeError("Timed out waiting for server handshake") from exc
            if isinstance(first, (bytes, bytearray)) and first[:1] == b"\x00":
                print("Handshake received.", flush=True)
                return
            if isinstance(first, (bytes, bytearray)) and first[:1] in (b"\x01", b"\x02"):
                self._early_messages.append(first)
                print("Handshake received.", flush=True)
                return
        raise RuntimeError("Stopped before handshake completed")

    async def _wait_for_primed(
        self,
        ws: websockets.ClientConnection,
        history_duration_sec: float = 0.0,
    ) -> None:
        timeout = max(300.0, history_duration_sec * 2.0 + 120.0)
        started = time.monotonic()
        bar = tqdm(
            total=1,
            desc="Server prefill",
            bar_format="{desc}: {elapsed}",
            leave=True,
        )

        async def refresh_elapsed() -> None:
            while True:
                elapsed = time.monotonic() - started
                bar.set_description(f"Server prefill ({elapsed:.0f}s)")
                bar.refresh()
                await asyncio.sleep(0.5)

        refresh_task = asyncio.create_task(refresh_elapsed())
        # Messages seen before PRIMED are held aside rather than pushed back
        # onto the queue this loop is draining, which would spin forever.
        deferred: list[bytes | str] = []
        try:
            while not self._closed:
                try:
                    message = await asyncio.wait_for(self._recv_next(ws), timeout=timeout)
                except TimeoutError as exc:
                    raise RuntimeError(
                        "Timed out waiting for primed acknowledgement"
                    ) from exc
                if not isinstance(message, (bytes, bytearray)) or not message:
                    continue
                if message[0] != MSG_PRIMED:
                    deferred.append(message)
                    continue
                self._early_messages[:0] = deferred
                payload = json.loads(message[1:].decode("utf-8"))
                bar.set_description("Server prefill")
                bar.update(1)
                print(
                    "Primed. "
                    f"history={payload.get('history_duration_sec', '?')}s "
                    f"elapsed={payload.get('elapsed_sec', '?')}s "
                    f"rt_factor={payload.get('realtime_factor', '?')}x "
                    f"truncated_frames={payload.get('truncated_frames', 0)}",
                    flush=True,
                )
                await self._signal_queue.put(
                    ModelSignal(
                        kind="duplex_primed",
                        received_at=time.monotonic(),
                        data=payload,
                    )
                )
                return
            raise RuntimeError("Stopped before primed acknowledgement")
        finally:
            refresh_task.cancel()
            await asyncio.gather(refresh_task, return_exceptions=True)
            bar.close()

    async def send_audio(self, pcm: np.ndarray) -> None:
        if self._closed or self._ws is None:
            return
        if pcm.size == 0:
            return
        if self._metrics.first_audio_sent_at is None:
            self._metrics.first_audio_sent_at = time.monotonic()
        self._opus.append_pcm(pcm)

    async def _send_loop(self) -> None:
        assert self._ws is not None
        while not self._closed:
            await asyncio.sleep(0.001)
            msg = self._opus.drain_encoded()
            if msg:
                # send_audio only fills the Opus writer, so this is the only
                # place a dead socket is observable while streaming.
                try:
                    await self._ws.send(b"\x01" + msg)
                except websockets.ConnectionClosed as error:
                    self._io_error = error
                    return

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        while not self._closed:
            if self._early_messages:
                msg = self._early_messages.pop(0)
            else:
                try:
                    msg = await self._ws.recv()
                except websockets.ConnectionClosed as error:
                    if not self._closed:
                        self._io_error = error
                    break
            if not msg or msg[0] not in (1, 2):
                continue
            kind, payload = msg[0], msg[1:]
            if kind == 1:
                if self._metrics.first_audio_received_at is None:
                    self._metrics.first_audio_received_at = time.monotonic()
                    if self._metrics.first_audio_sent_at is not None:
                        delta = (
                            self._metrics.first_audio_received_at
                            - self._metrics.first_audio_sent_at
                        )
                        self._metrics.ttfb_ms = delta * 1000
                self._opus.append_encoded(payload)
                await self._drain_opus_frames()
            else:
                await self._text_queue.put(payload.decode(errors="ignore"))
        await self._flush_opus_frames()
        await self._audio_queue.put(None)
        await self._text_queue.put(None)

    async def _drain_opus_frames(self) -> None:
        while not self._closed:
            pcm = self._opus.read_pcm_frame()
            if pcm.size == 0:
                return
            for frame in self._frames.push(pcm):
                await self._audio_queue.put(AudioFrame(frame))

    async def _flush_opus_frames(self) -> None:
        remainder = self._frames.flush()
        if remainder.size:
            await self._audio_queue.put(AudioFrame(remainder))

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

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = [task for task in (self._send_task, self._recv_task) if task is not None]
        for task in tasks:
            task.cancel()
        # Gathering keeps an I/O failure from surfacing later as an unretrieved
        # task exception instead of as a failed rollout.
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                self._io_error = self._io_error or result
        if self._ws is not None:
            await self._ws.close()
        await self._audio_queue.put(None)
        await self._text_queue.put(None)
        await self._signal_queue.put(None)


def create(config: ModelConfig) -> PersonaPlexModel:
    return PersonaPlexModel(config)
