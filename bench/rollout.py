"""Deterministic recorded-audio rollouts for realtime models."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from bench.audio import (
    resample,
    write_wav_pcm16,
)
from bench.points import HOLD_DRAIN, RolloutScenario, select_context
from bench.protocol import (
    AudioEvent,
    AudioFrame,
    AudioInterrupted,
    ContextType,
    ModelSignal,
    RealtimeModel,
    SessionConfig,
)
from bench.registry import ModelConfig
from bench.session import RealtimeSession
from bench.speech import (
    SPEECH_MIN_DURATION_SEC,
    SPEECH_RMS_THRESHOLD,
    PlayedAudio,
    last_speech_end_sec,
)
from bench.trace import RolloutTrace, TraceEvent

PAUSE_DRAIN_SEC = 3.0
PAUSE_NO_AUDIO_EXTENSION_SEC = 8.0
PAUSE_ONSET_TAIL_SEC = 1.0
EOT_TRAILING_SILENCE_SEC = 1.5
EOT_NO_SPEECH_SEC = 8.0
EOT_HARD_CAP_SEC = 30.0
EOT_POLL_SEC = 0.05


class TraceSink:
    def __init__(self, sample_rate: int, block_samples: int):
        self.sample_rate = sample_rate
        self.block_samples = block_samples
        self.started_at = 0.0
        self.events: list[TraceEvent] = []
        self.played_audio: list[PlayedAudio] = []
        self.text_parts: list[str] = []
        self.played_pcm: list[np.ndarray] = []
        self.sent_pcm: list[np.ndarray] = []
        self.generation = 0
        self.queue: asyncio.Queue[tuple[int, AudioFrame, float] | None] = (
            asyncio.Queue()
        )
        self.playback_task: asyncio.Task | None = None

    def start(self) -> None:
        self.started_at = time.monotonic()
        self.playback_task = asyncio.create_task(self.playback())

    def relative_time(self, timestamp: float) -> float:
        return max(0.0, timestamp - self.started_at)

    @property
    def has_model_audio(self) -> bool:
        """Whether the model has produced any audio, played or still queued."""
        return bool(self.played_audio) or not self.queue.empty()

    async def receive_audio(self, event: AudioEvent, received_at: float) -> None:
        at_sec = self.relative_time(received_at)
        if isinstance(event, AudioInterrupted):
            self.generation += 1
            while not self.queue.empty():
                self.queue.get_nowait()
            self.events.append(TraceEvent(kind="audio_interrupted", at_sec=at_sec))
            return

        duration = event.pcm.shape[0] / self.sample_rate
        self.events.append(
            TraceEvent(
                kind="model_audio_received",
                at_sec=at_sec,
                duration_sec=duration,
            )
        )
        await self.queue.put((self.generation, event, received_at))

    async def receive_text(self, text: str, received_at: float) -> None:
        self.text_parts.append(text)
        self.events.append(
            TraceEvent(
                kind="model_text",
                at_sec=self.relative_time(received_at),
                data={"text": text},
            )
        )

    async def receive_signal(self, signal: ModelSignal) -> None:
        self.events.append(
            TraceEvent(
                kind=signal.kind,
                at_sec=self.relative_time(signal.received_at),
                data=signal.data,
            )
        )

    async def playback(self) -> None:
        # Blocks are timed from when the audio arrived and from the previous
        # block's scheduled end, never from the clock this loop just slept
        # against: asyncio.sleep overshoots every block by about a millisecond,
        # and re-anchoring on it drifts playback several percent slow. Pause
        # scoring compares these timestamps with input_audio_sent, so the two
        # timelines have to stay on the same clock.
        next_block_at = 0.0
        while True:
            queued = await self.queue.get()
            if queued is None:
                return
            generation, frame, received_at = queued
            offset = 0
            while offset < frame.pcm.shape[0] and generation == self.generation:
                count = min(self.block_samples, frame.pcm.shape[0] - offset)
                pcm = frame.pcm[offset : offset + count].astype(np.float32, copy=True)
                duration = count / self.sample_rate
                started_at = max(received_at, next_block_at)
                next_block_at = started_at + duration
                start_sec = self.relative_time(started_at)
                rms = float(np.sqrt(np.mean(pcm * pcm))) if pcm.size else 0.0
                self.played_pcm.append(pcm)
                self.played_audio.append(
                    PlayedAudio(
                        start_sec=start_sec,
                        end_sec=start_sec + duration,
                        rms=rms,
                    )
                )
                frame.played(count)
                offset += count
                await asyncio.sleep(max(0.0, next_block_at - time.monotonic()))

    async def stop(self) -> None:
        await self.queue.put(None)
        if self.playback_task is not None:
            await self.playback_task

    def response_pcm(self) -> np.ndarray:
        if not self.played_pcm:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.played_pcm)

    def streamed_pcm(self) -> np.ndarray:
        if not self.sent_pcm:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.sent_pcm)


def render_rollout_pcm(
    input_pcm: np.ndarray,
    input_sample_rate: int,
    sink: TraceSink,
) -> np.ndarray:
    user_pcm = resample(input_pcm, input_sample_rate, sink.sample_rate)
    model_end = max(
        (
            int(round(audio.end_sec * sink.sample_rate))
            for audio in sink.played_audio
        ),
        default=0,
    )
    length = max(user_pcm.shape[0], model_end)
    stereo = np.zeros((length, 2), dtype=np.float32)
    stereo[: user_pcm.shape[0], 0] = user_pcm
    for audio, pcm in zip(sink.played_audio, sink.played_pcm, strict=True):
        start = int(round(audio.start_sec * sink.sample_rate))
        end = min(start + pcm.shape[0], length)
        stereo[start:end, 1] = pcm[: end - start]
    return stereo


def apply_session_config(
    session_config: SessionConfig, scenario: RolloutScenario
) -> SessionConfig:
    if scenario.detect_turns:
        return session_config
    return replace(session_config, detect_turns=False)


async def stream_pcm(
    session: RealtimeSession,
    pcm: np.ndarray,
    frame_samples: int,
    sample_rate: int,
    sink: TraceSink,
) -> int:
    frame_duration = frame_samples / sample_rate
    frame_index = 0
    for offset in range(0, pcm.shape[0], frame_samples):
        frame = pcm[offset : offset + frame_samples]
        if frame.shape[0] < frame_samples:
            frame = np.pad(frame, (0, frame_samples - frame.shape[0]))
        await send_input_frame(session, sink, frame, frame_index, frame_duration)
        frame_index += 1
    return frame_index


async def send_input_frame(
    session: RealtimeSession,
    sink: TraceSink,
    frame: np.ndarray,
    frame_index: int,
    frame_duration: float,
) -> None:
    # scheduled_at_sec is the source-time *end* of the frame: the frame is
    # sent once all of its audio has elapsed. observed_input_time relies
    # on this to map source time to wall clock.
    scheduled_at = (frame_index + 1) * frame_duration
    deadline = sink.started_at + scheduled_at
    await asyncio.sleep(max(0.0, deadline - time.monotonic()))
    await session.send_audio(frame.astype(np.float32, copy=False))
    sink.sent_pcm.append(frame.astype(np.float32, copy=True))
    sink.events.append(
        TraceEvent(
            kind="input_audio_sent",
            at_sec=sink.relative_time(time.monotonic()),
            duration_sec=frame_duration,
            data={"scheduled_at_sec": scheduled_at},
        )
    )


@dataclass(frozen=True)
class DrainConfig:
    pause_drain_sec: float = PAUSE_DRAIN_SEC
    pause_no_audio_extension_sec: float = PAUSE_NO_AUDIO_EXTENSION_SEC
    pause_onset_tail_sec: float = PAUSE_ONSET_TAIL_SEC
    eot_trailing_silence_sec: float = EOT_TRAILING_SILENCE_SEC
    eot_no_speech_sec: float = EOT_NO_SPEECH_SEC
    eot_hard_cap_sec: float = EOT_HARD_CAP_SEC


def silence_frame_count(
    duration_sec: float, sample_rate: int, frame_samples: int
) -> int:
    return int(np.ceil(duration_sec * sample_rate / frame_samples))


async def drain_live_turn(
    session: RealtimeSession,
    sink: TraceSink,
    scenario: RolloutScenario,
    frame_samples: int,
    frame_index: int,
    drain: DrainConfig,
    *,
    step_driven: bool,
) -> bool:
    """Drain after the live input: pause holds are shared, EOT forks by family.

    A step-driven duplex model (PersonaPlex, Moshi) runs one LM step per
    received frame, so its EOT drain streams silence to keep it stepping. A
    hosted model's VAD/clock runs server-side, so its EOT drain commits the
    buffer and waits on the wall clock. The pause hold streams real silence
    frames for both families.
    """
    if scenario.drain == HOLD_DRAIN:
        await stream_hold_drain(
            session, sink, scenario, frame_samples, frame_index, drain
        )
        return False
    if step_driven:
        timed_out = await stream_silence_until_speech_done(
            session,
            sink,
            frame_samples,
            scenario.sample_rate,
            frame_index,
            sink.relative_time(time.monotonic()),
            trailing_silence_sec=drain.eot_trailing_silence_sec,
            no_speech_sec=drain.eot_no_speech_sec,
            hard_cap_sec=drain.eot_hard_cap_sec,
        )
    else:
        await session.end_input()
        timed_out = await wait_until_speech_done(
            sink,
            sink.relative_time(time.monotonic()),
            trailing_silence_sec=drain.eot_trailing_silence_sec,
            no_speech_sec=drain.eot_no_speech_sec,
            hard_cap_sec=drain.eot_hard_cap_sec,
        )
    record_eot_drain(sink, timed_out)
    return timed_out


def record_eot_drain(sink: TraceSink, timed_out: bool) -> None:
    sink.events.append(
        TraceEvent(
            kind="eot_drain_done",
            at_sec=sink.relative_time(time.monotonic()),
            data={"timed_out": timed_out},
        )
    )


def speech_done(
    sink: TraceSink,
    handoff_sec: float,
    *,
    trailing_silence_sec: float,
    no_speech_sec: float,
    rms_threshold: float = SPEECH_RMS_THRESHOLD,
    min_duration_sec: float = SPEECH_MIN_DURATION_SEC,
) -> bool:
    """Whether the model has finished responding, on the same speech definition
    scoring uses. Audio still queued for playback counts as not finished."""
    if not sink.queue.empty():
        return False
    now = sink.relative_time(time.monotonic())
    speech_end = last_speech_end_sec(
        sink.played_audio,
        handoff_sec,
        rms_threshold=rms_threshold,
        min_duration_sec=min_duration_sec,
    )
    if speech_end is not None:
        return now - speech_end >= trailing_silence_sec
    return now - handoff_sec >= no_speech_sec


async def wait_until_speech_done(
    sink: TraceSink,
    handoff_sec: float,
    *,
    trailing_silence_sec: float = EOT_TRAILING_SILENCE_SEC,
    no_speech_sec: float = EOT_NO_SPEECH_SEC,
    hard_cap_sec: float = EOT_HARD_CAP_SEC,
    poll_sec: float = EOT_POLL_SEC,
) -> bool:
    deadline = time.monotonic() + hard_cap_sec
    while time.monotonic() < deadline:
        if speech_done(
            sink,
            handoff_sec,
            trailing_silence_sec=trailing_silence_sec,
            no_speech_sec=no_speech_sec,
        ):
            return False
        await asyncio.sleep(poll_sec)
    return True


async def stream_hold_drain(
    session: RealtimeSession,
    sink: TraceSink,
    scenario: RolloutScenario,
    frame_samples: int,
    frame_index: int,
    drain: DrainConfig,
) -> None:
    """Drain a pause hold as real silence frames for both model families.

    The live pause input already runs through pause_end, so the scored window
    was streamed. A step-driven model needs the drain frames to keep being
    stepped; a hosted model's server VAD only ever closes the user turn on
    silence it receives, so a wall-clock wait here means no response is
    created at all.
    """
    count = silence_frame_count(
        drain.pause_drain_sec, scenario.sample_rate, frame_samples
    )
    frame_index = await stream_silence_frames(
        session, sink, frame_samples, scenario.sample_rate, frame_index, count
    )
    # A live model's first audio can still be in flight when the fixed drain
    # ends: grok often commits its response late in the hold and takes ~0.6s
    # (healthy) to first audio, which was recorded as no_model_audio. A model
    # that has produced nothing yet gets a bounded extension, and once audio
    # appears a short tail is streamed so the onset clears the speech gate.
    # Audio that already arrived inside the fixed drain gets no tail: the
    # onset sits inside the drain frames that were streamed around it.
    if sink.has_model_audio:
        return
    silence = np.zeros(frame_samples, dtype=np.float32)
    frame_duration = frame_samples / scenario.sample_rate
    extension = silence_frame_count(
        drain.pause_no_audio_extension_sec, scenario.sample_rate, frame_samples
    )
    for _ in range(extension):
        await send_input_frame(session, sink, silence, frame_index, frame_duration)
        frame_index += 1
        if sink.has_model_audio:
            tail = silence_frame_count(
                drain.pause_onset_tail_sec, scenario.sample_rate, frame_samples
            )
            await stream_silence_frames(
                session, sink, frame_samples, scenario.sample_rate, frame_index, tail
            )
            return


async def stream_silence_frames(
    session: RealtimeSession,
    sink: TraceSink,
    frame_samples: int,
    sample_rate: int,
    frame_index: int,
    count: int,
) -> int:
    silence = np.zeros(frame_samples, dtype=np.float32)
    frame_duration = frame_samples / sample_rate
    for _ in range(count):
        await send_input_frame(session, sink, silence, frame_index, frame_duration)
        frame_index += 1
    return frame_index


async def stream_silence_until_speech_done(
    session: RealtimeSession,
    sink: TraceSink,
    frame_samples: int,
    sample_rate: int,
    frame_index: int,
    handoff_sec: float,
    *,
    trailing_silence_sec: float = EOT_TRAILING_SILENCE_SEC,
    no_speech_sec: float = EOT_NO_SPEECH_SEC,
    hard_cap_sec: float = EOT_HARD_CAP_SEC,
) -> bool:
    """Drain a step-driven duplex model by keeping it fed with silence.

    Such a model runs one LM step per received frame, so a wall-clock wait
    leaves it unable to produce the very response being measured. It is sent
    frame-aligned silence on stream_pcm's schedule until it stops speaking.
    """
    silence = np.zeros(frame_samples, dtype=np.float32)
    frame_duration = frame_samples / sample_rate
    deadline = time.monotonic() + hard_cap_sec
    while time.monotonic() < deadline:
        await send_input_frame(session, sink, silence, frame_index, frame_duration)
        frame_index += 1
        if speech_done(
            sink,
            handoff_sec,
            trailing_silence_sec=trailing_silence_sec,
            no_speech_sec=no_speech_sec,
        ):
            return False
    return True


async def run_rollout(
    model: RealtimeModel,
    model_config: ModelConfig,
    session_config: SessionConfig,
    scenario: RolloutScenario,
    output_dir: Path,
    rollout_index: int = 0,
    pause_drain_sec: float = PAUSE_DRAIN_SEC,
    pause_no_audio_extension_sec: float = PAUSE_NO_AUDIO_EXTENSION_SEC,
    pause_onset_tail_sec: float = PAUSE_ONSET_TAIL_SEC,
    eot_trailing_silence_sec: float = EOT_TRAILING_SILENCE_SEC,
    eot_no_speech_sec: float = EOT_NO_SPEECH_SEC,
    eot_hard_cap_sec: float = EOT_HARD_CAP_SEC,
) -> RolloutTrace:
    if scenario.sample_rate != model_config.audio.input_sample_rate:
        raise ValueError(
            f"scenario sample rate {scenario.sample_rate} does not match "
            f"model input rate {model_config.audio.input_sample_rate}"
        )

    context = select_context(
        scenario.contexts, model.supported_contexts, point_id=scenario.point_id
    )
    step_driven = context.kind is ContextType.DUPLEX_AUDIO
    session_config = apply_session_config(session_config, scenario)
    sink = TraceSink(
        model_config.audio.output_sample_rate,
        min(480, model_config.audio.frame_samples),
    )
    session = RealtimeSession(
        model,
        session_config,
        on_audio=sink.receive_audio,
        on_text=sink.receive_text,
        on_signal=sink.receive_signal,
    )
    drain = DrainConfig(
        pause_drain_sec=pause_drain_sec,
        pause_no_audio_extension_sec=pause_no_audio_extension_sec,
        pause_onset_tail_sec=pause_onset_tail_sec,
        eot_trailing_silence_sec=eot_trailing_silence_sec,
        eot_no_speech_sec=eot_no_speech_sec,
        eot_hard_cap_sec=eot_hard_cap_sec,
    )
    live_pcm = scenario.live_pcm(step_driven=step_driven)
    timed_out = False
    try:
        await session.start(context)
        sink.start()
        frame_index = await stream_pcm(
            session,
            live_pcm,
            model_config.audio.frame_samples,
            scenario.sample_rate,
            sink,
        )
        timed_out = await drain_live_turn(
            session,
            sink,
            scenario,
            model_config.audio.frame_samples,
            frame_index,
            drain,
            step_driven=step_driven,
        )
    finally:
        try:
            await session.close()
        finally:
            await sink.stop()

    # A transport that died mid-rollout still produces a complete-looking trace,
    # so surface it rather than scoring the silence that followed. Checked after
    # close, which is where a failed I/O task is finally collected.
    connection_error = model.connection_error
    if connection_error is not None:
        raise connection_error

    response_pcm = sink.response_pcm()
    trace = RolloutTrace(
        point_id=scenario.point_id,
        model_id=model_config.id,
        rollout_index=rollout_index,
        expected_action=scenario.expected_action,
        checkpoint_sec=scenario.checkpoint_sec,
        window_end_sec=scenario.window_end_sec,
        events=sink.events,
        played_audio=sink.played_audio,
        text="".join(sink.text_parts),
        response_audio=response_pcm,
        sample_rate=model_config.audio.output_sample_rate,
        session=asdict(session_config),
        timed_out=timed_out,
        played_pcm=list(sink.played_pcm),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_wav_pcm16(output_dir / "response.wav", response_pcm, trace.sample_rate)
    rollout_pcm = render_rollout_pcm(
        sink.streamed_pcm(),
        scenario.sample_rate,
        sink,
    )
    write_wav_pcm16(output_dir / "rollout.wav", rollout_pcm, trace.sample_rate)
    with (output_dir / "trace.json").open("w", encoding="utf-8") as handle:
        json.dump(trace.to_dict(), handle, indent=2)
        handle.write("\n")
    return trace
