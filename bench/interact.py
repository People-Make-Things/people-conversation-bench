"""Live mic/speaker session for any RealtimeModel adapter."""

from __future__ import annotations

import asyncio
import queue
import threading
import time

import numpy as np
import sounddevice as sd
import websockets.exceptions as wsex

from bench.points import load_point, select_context
from bench.protocol import (
    AudioFrame,
    AudioInterrupted,
    Context,
    PreloadNotSupportedError,
    RealtimeModel,
    SessionConfig,
)
from bench.registry import ModelConfig, load_model
from bench.session import RealtimeSession


class TurnTextPrinter:
    """Buffer streamed tokens and print one line per model speaking turn."""

    def __init__(self, idle_seconds: float = 0.8):
        self.idle_seconds = idle_seconds
        self.buffer = ""
        self.last_token_at: float | None = None

    def add(self, token: str) -> None:
        cleaned = token.replace("▁", " ")
        if cleaned.strip() == "" and not self.buffer:
            return
        self.buffer += cleaned
        self.last_token_at = time.monotonic()

    def flush(self) -> None:
        line = self.buffer.strip()
        if line:
            print(f"Model: {line}", flush=True)
        self.buffer = ""
        self.last_token_at = None

    def maybe_flush_idle(self) -> None:
        if not self.buffer or self.last_token_at is None:
            return
        if time.monotonic() - self.last_token_at >= self.idle_seconds:
            self.flush()


class InteractSession:
    def __init__(
        self,
        model: RealtimeModel,
        config: ModelConfig,
        session: SessionConfig,
        input_device: int | None,
        output_device: int | None,
    ):
        self.model = model
        self.config = config
        self.session = session
        self.done = False
        self.output_interrupted = threading.Event()
        self.text_printer = TurnTextPrinter()
        self.output_queue: queue.Queue[AudioFrame] = queue.Queue()
        self.realtime = RealtimeSession(
            model,
            session,
            on_audio=self.handle_audio,
            on_text=self.handle_text,
        )

        frame_samples = config.audio.frame_samples
        sample_rate = config.audio.input_sample_rate
        device_block = min(480, frame_samples)

        stream_kwargs = {
            "samplerate": sample_rate,
            "channels": 1,
            "blocksize": device_block,
            "dtype": "float32",
            "latency": "low",
        }
        if input_device is not None:
            stream_kwargs["device"] = input_device

        out_kwargs = {
            "samplerate": config.audio.output_sample_rate,
            "channels": 1,
            "blocksize": device_block,
            "dtype": "float32",
            "latency": "low",
        }
        if output_device is not None:
            out_kwargs["device"] = output_device

        self.frame_samples = frame_samples
        self.device_block = device_block
        self._input_pending = np.zeros(0, dtype=np.float32)
        self._output_frame: AudioFrame | None = None
        self._output_offset = 0
        self._mic_heard = False
        self._model_heard = False
        self.in_stream = sd.InputStream(callback=self.on_audio_input, **stream_kwargs)
        self.out_stream = sd.OutputStream(callback=self.on_audio_output, **out_kwargs)

    def submit_audio(self, pcm: np.ndarray) -> None:
        future = asyncio.run_coroutine_threadsafe(self.realtime.send_audio(pcm), self.loop)

        def done(handle: asyncio.Future) -> None:
            exc = handle.exception()
            if exc is not None:
                print(f"[ERROR] send_audio: {exc}", flush=True)

        future.add_done_callback(done)

    def on_audio_input(self, in_data, frames, time_info, status):
        if status:
            print("[WARN] mic:", status, flush=True)
        if self.done:
            return

        pcm = in_data[:, 0].astype(np.float32, copy=False)
        rms = float(np.sqrt(np.mean(pcm * pcm)))
        if rms > 0.01 and not self._mic_heard:
            self._mic_heard = True
            print("Mic input detected.", flush=True)

        self._input_pending = np.concatenate([self._input_pending, pcm])
        while self._input_pending.shape[0] >= self.frame_samples:
            chunk = self._input_pending[: self.frame_samples].astype(np.float32, copy=True)
            self._input_pending = self._input_pending[self.frame_samples :]
            self.submit_audio(chunk)

    def on_audio_output(self, out_data, frames, time_info, status):
        if status:
            print("[WARN] speaker:", status, flush=True)
        if self.output_interrupted.is_set():
            self.output_interrupted.clear()
            self._output_frame = None
            self._output_offset = 0
            out_data.fill(0)
            return

        out_data.fill(0)
        written = 0
        while written < out_data.shape[0]:
            if self._output_frame is None:
                try:
                    self._output_frame = self.output_queue.get(block=False)
                except queue.Empty:
                    break
                self._output_offset = 0

            frame = self._output_frame
            available = frame.pcm.shape[0] - self._output_offset
            copied = min(available, out_data.shape[0] - written)
            out_data[written : written + copied, 0] = np.clip(
                frame.pcm[self._output_offset : self._output_offset + copied],
                -1.0,
                1.0,
            )
            frame.played(copied)
            written += copied
            self._output_offset += copied
            if self._output_offset == frame.pcm.shape[0]:
                self._output_frame = None
                self._output_offset = 0

    def interrupt_playback(self) -> None:
        self.output_interrupted.set()
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except queue.Empty:
                break

    async def handle_audio(self, event, received_at: float) -> None:
        if isinstance(event, AudioInterrupted):
            self.interrupt_playback()
            return
        if not self._model_heard:
            self._model_heard = True
            print("Model audio received.", flush=True)
        self.output_queue.put(event)

    async def handle_text(self, token: str, received_at: float) -> None:
        self.text_printer.add(token)

    async def text_flush_loop(self) -> None:
        while not self.done:
            await asyncio.sleep(0.1)
            self.text_printer.maybe_flush_idle()

    async def mic_watchdog(self) -> None:
        await asyncio.sleep(3)
        if self.done or self._mic_heard:
            return
        print(
            "[WARN] No mic activity detected. Grant microphone access to your terminal app "
            "and verify the device with --list-devices.",
            flush=True,
        )

    async def run(
        self,
        preload_point: str | None = None,
        context: Context | None = None,
    ) -> None:
        self.loop = asyncio.get_running_loop()
        tasks = []
        try:
            if context is not None:
                if context.kind not in self.model.supported_contexts:
                    supported = [kind.value for kind in self.model.supported_contexts]
                    raise PreloadNotSupportedError(
                        f"Model supports {supported} but received "
                        f"{context.kind.value} context"
                    )
                source = f" from {preload_point}" if preload_point else ""
                print(
                    f"Preloading {context.kind.value} context{source}",
                    flush=True,
                )
            await self.realtime.start(context)
            print("Connected. Speak into your microphone. Press Ctrl+C to stop.\n")

            with self.in_stream, self.out_stream:
                tasks = [
                    asyncio.create_task(self.text_flush_loop()),
                    asyncio.create_task(self.mic_watchdog()),
                ]
                await asyncio.Future()
        finally:
            self.done = True
            self.text_printer.flush()
            for task in tasks:
                task.cancel()
            await self.realtime.close()
            await asyncio.gather(*tasks, return_exceptions=True)


def build_session_config(
    config: ModelConfig,
    voice_prompt: str | None,
    text_prompt: str | None,
    seed: int | None,
) -> SessionConfig:
    defaults = config.session
    resolved_voice_prompt = voice_prompt if voice_prompt is not None else defaults.voice_prompt
    resolved_text_prompt = text_prompt if text_prompt is not None else defaults.text_prompt

    resolved_seed = defaults.seed
    if seed is not None:
        resolved_seed = None if seed == -1 else seed

    extra = dict(defaults.extra or {})
    return SessionConfig(
        voice_prompt=resolved_voice_prompt,
        text_prompt=resolved_text_prompt,
        seed=resolved_seed,
        extra=extra,
    )


def format_audio_device(index: int) -> str:
    info = sd.query_devices(index)
    return f"[{index}] {info['name']}"


def run(
    model: str,
    voice_prompt: str | None = None,
    text_prompt: str | None = None,
    seed: int | None = None,
    input_device: int | None = None,
    output_device: int | None = None,
    list_devices: bool = False,
    preload_point: str | None = None,
    context: Context | None = None,
) -> None:
    if list_devices:
        print(sd.query_devices())
        return

    config, adapter = load_model(model)
    selected_context = context
    resolved_text_prompt = text_prompt
    if preload_point is not None:
        if selected_context is not None:
            raise ValueError("pass only one of preload_point or context")
        point = load_point(preload_point, kinds=adapter.supported_contexts)
        selected_context = select_context(
            point.contexts, adapter.supported_contexts, point_id=point.point_id
        )
        if resolved_text_prompt is None:
            resolved_text_prompt = point.text_prompt
    session = build_session_config(config, voice_prompt, resolved_text_prompt, seed)

    print(f"Model: {config.name} ({config.id})")
    if session.text_prompt:
        print(f"Text prompt: {session.text_prompt}")
    else:
        print("Text prompt: (none)")
    print(f"Voice prompt: {session.voice_prompt}")

    in_idx = input_device
    out_idx = output_device
    if in_idx is None or out_idx is None:
        default_in, default_out = sd.default.device
        if in_idx is None:
            in_idx = default_in
        if out_idx is None:
            out_idx = default_out
    print(f"Microphone: {format_audio_device(in_idx)}")
    print(f"Speaker:    {format_audio_device(out_idx)}")
    print("If audio is silent, rerun with --list-devices and set --input-device/--output-device.\n")

    interact = InteractSession(
        adapter,
        config,
        session,
        input_device,
        output_device,
    )

    try:
        asyncio.run(
            interact.run(
                preload_point=preload_point,
                context=selected_context,
            )
        )
    except wsex.ConnectionClosedError as exc:
        print(f"[WARN] connection closed: {exc}")
    except KeyboardInterrupt:
        pass
