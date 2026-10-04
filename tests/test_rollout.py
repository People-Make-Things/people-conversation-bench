from __future__ import annotations

import asyncio
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from bench.points import EotScenario, PauseScenario, ReferenceResponse
from bench.protocol import (
    AudioEvent,
    AudioFrame,
    Context,
    ContextType,
    DuplexAudioContext,
    RealtimeModel,
    SessionConfig,
    SessionMetrics,
    SessionMode,
    event_context,
)
from bench.registry import AudioConfig, ConnectionConfig, ModelConfig
from bench.rollout import (
    TraceSink,
    run_rollout,
    silence_frame_count,
    wait_until_speech_done,
)
from bench.speech import PlayedAudio


class FakeModel(RealtimeModel):
    def __init__(self):
        self.audio: asyncio.Queue[AudioEvent | None] = asyncio.Queue()
        self.text: asyncio.Queue[str | None] = asyncio.Queue()
        self.sent_response = False
        self.sent_frames = 0
        self.ended_input = False
        self.session_metrics = SessionMetrics()

    @property
    def supported_contexts(self) -> tuple[ContextType, ...]:
        return (ContextType.TEXT,)

    def endpoint(self, session: SessionConfig) -> str:
        return "fake"

    async def connect(
        self,
        session: SessionConfig,
        *,
        mode: SessionMode = SessionMode.LIVE,
    ) -> None:
        pass

    async def preload_context(self, context: Context) -> None:
        pass

    async def send_audio(self, pcm: np.ndarray) -> None:
        self.sent_frames += 1
        if not self.sent_response:
            self.sent_response = True
            await self.audio.put(
                AudioFrame(np.full(480, 0.1, dtype=np.float32))
            )

    async def end_input(self) -> None:
        self.ended_input = True

    async def recv_audio(self) -> AsyncIterator[AudioEvent]:
        while True:
            event = await self.audio.get()
            if event is None:
                return
            yield event

    async def recv_text(self) -> AsyncIterator[str]:
        while True:
            text = await self.text.get()
            if text is None:
                return
            yield text

    async def close(self) -> None:
        await self.audio.put(None)
        await self.text.put(None)

    @property
    def metrics(self) -> SessionMetrics:
        return self.session_metrics


def test_hosted_pause_streams_real_frames_through_the_hold(tmp_path: Path) -> None:
    """A hosted model's server VAD only closes the user turn on silence it
    receives, so the pause path must stream the pause_end audio and then real
    drain frames rather than going quiet at pause_start and waiting."""

    async def run() -> None:
        config = ModelConfig(
            id="fake",
            name="Fake",
            model_dir=tmp_path,
            adapter_path=tmp_path / "adapter.py",
            connection=ConnectionConfig(url="fake"),
            audio=AudioConfig(
                input_sample_rate=24000,
                output_sample_rate=24000,
                encoding="pcm16",
                frame_samples=480,
            ),
            session=SessionConfig(),
        )
        scenario = PauseScenario(
            point_id="point",
            point_dir=tmp_path,
            contexts={ContextType.TEXT: event_context([])},
            input_pcm=np.full(480, 0.05, dtype=np.float32),
            # pause_end audio: speech frame plus the real pause silence.
            pause_input_pcm=np.concatenate(
                [np.full(480, 0.05, dtype=np.float32), np.zeros(480, dtype=np.float32)]
            ),
            sample_rate=24000,
            checkpoint_sec=0.02,
            window_end_sec=0.04,
        )

        model = FakeModel()
        trace = await run_rollout(
            model,
            config,
            SessionConfig(),
            scenario,
            tmp_path / "result",
            pause_drain_sec=0.01,
        )

        # Two frames of pause_end audio, then one frame of drain silence, all
        # on stream_pcm's schedule so observed_input_time keeps mapping.
        assert model.sent_frames == 3
        scheduled = [
            event.data["scheduled_at_sec"]
            for event in trace.events
            if event.kind == "input_audio_sent"
        ]
        assert scheduled == pytest.approx([0.02, 0.04, 0.06])
        assert trace.session["detect_turns"] is True

    asyncio.run(run())


class DeadSocketModel(FakeModel):
    """Accepts everything, produces nothing, and reports the socket died."""

    def __init__(self):
        super().__init__()
        self.failure = ConnectionResetError("socket closed mid-rollout")

    @property
    def connection_error(self) -> BaseException | None:
        return self.failure

    async def send_audio(self, pcm: np.ndarray) -> None:
        self.sent_frames += 1


def test_rollout_fails_loudly_when_the_transport_died(tmp_path: Path) -> None:
    """A dead socket still yields a full set of input_audio_sent events for
    frames that never left, so the trace alone looks like a healthy rollout."""

    async def run() -> None:
        config = ModelConfig(
            id="fake",
            name="Fake",
            model_dir=tmp_path,
            adapter_path=tmp_path / "adapter.py",
            connection=ConnectionConfig(url="fake"),
            audio=AudioConfig(
                input_sample_rate=24000,
                output_sample_rate=24000,
                encoding="pcm16",
                frame_samples=480,
            ),
            session=SessionConfig(),
        )
        scenario = PauseScenario(
            point_id="point",
            point_dir=tmp_path,
            contexts={ContextType.TEXT: event_context([])},
            input_pcm=np.full(480, 0.05, dtype=np.float32),
            pause_input_pcm=np.zeros(480, dtype=np.float32),
            sample_rate=24000,
            checkpoint_sec=0.02,
            window_end_sec=0.02,
        )

        with pytest.raises(ConnectionResetError):
            await run_rollout(
                DeadSocketModel(),
                config,
                SessionConfig(),
                scenario,
                tmp_path / "result",
                pause_drain_sec=0.01,
                pause_no_audio_extension_sec=0.02,
            )

    asyncio.run(run())


def test_playback_timestamps_do_not_drift_behind_wall_clock() -> None:
    async def run() -> None:
        sink = TraceSink(24000, 480)
        sink.start()
        queued_sec = 2.0
        frame_sec = 0.08
        for _ in range(int(queued_sec / frame_sec)):
            frame = AudioFrame(np.zeros(1920, dtype=np.float32))
            await sink.receive_audio(frame, time.monotonic())
        await sink.stop()

        played_sec = sum(
            played.end_sec - played.start_sec for played in sink.played_audio
        )
        assert played_sec == pytest.approx(queued_sec, abs=1e-6)
        # Timestamps must describe the audio, not the scheduler: sleeping per
        # block accumulates roughly 5% of overshoot, which would land here.
        assert sink.played_audio[-1].end_sec == pytest.approx(queued_sec, abs=0.02)

    asyncio.run(run())


def test_playback_timestamps_preserve_silence_between_frames() -> None:
    async def run() -> None:
        sink = TraceSink(24000, 480)
        sink.start()
        gap_sec = 0.3
        await sink.receive_audio(
            AudioFrame(np.zeros(1920, dtype=np.float32)), time.monotonic()
        )
        await asyncio.sleep(gap_sec)
        await sink.receive_audio(
            AudioFrame(np.zeros(1920, dtype=np.float32)), time.monotonic()
        )
        await sink.stop()

        assert sink.played_audio[0].start_sec == pytest.approx(0.0, abs=0.02)
        blocks_per_frame = 1920 // 480
        after_gap = sink.played_audio[blocks_per_frame]
        assert after_gap.start_sec == pytest.approx(gap_sec, abs=0.05)

    asyncio.run(run())


def test_rollout_writes_model_response_wav(tmp_path: Path) -> None:
    async def run() -> None:
        config = ModelConfig(
            id="fake",
            name="Fake",
            model_dir=tmp_path,
            adapter_path=tmp_path / "adapter.py",
            connection=ConnectionConfig(url="fake"),
            audio=AudioConfig(
                input_sample_rate=24000,
                output_sample_rate=24000,
                encoding="pcm16",
                frame_samples=480,
            ),
            session=SessionConfig(),
        )
        scenario = PauseScenario(
            point_id="point",
            point_dir=tmp_path,
            contexts={ContextType.TEXT: event_context([])},
            input_pcm=np.full(480, 0.05, dtype=np.float32),
            pause_input_pcm=np.full(480, 0.05, dtype=np.float32),
            sample_rate=24000,
            checkpoint_sec=0.02,
            window_end_sec=0.02,
        )

        model = FakeModel()
        trace = await run_rollout(
            model,
            config,
            SessionConfig(),
            scenario,
            tmp_path / "result",
            pause_drain_sec=0.01,
        )

        # One frame of pause_end audio plus one frame of drain silence.
        assert model.sent_frames == 2
        input_event = next(
            event for event in trace.events if event.kind == "input_audio_sent"
        )
        assert input_event.data["scheduled_at_sec"] == 0.02
        assert trace.response_audio.shape == (480,)
        with wave.open(str(tmp_path / "result" / "response.wav"), "rb") as handle:
            assert handle.getnframes() == 480
        with wave.open(str(tmp_path / "result" / "rollout.wav"), "rb") as handle:
            assert handle.getnchannels() == 2
            samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
        stereo = samples.reshape(-1, 2)
        assert np.max(np.abs(stereo[:, 0])) > 0
        assert np.max(np.abs(stereo[:, 1])) > 0

    asyncio.run(run())


def test_wait_until_speech_done_after_trailing_silence() -> None:
    async def run() -> None:
        sink = TraceSink(24000, 480)
        sink.started_at = time.monotonic()
        sink.played_audio.append(PlayedAudio(0.0, 0.10, 0.1))
        timed_out = await wait_until_speech_done(
            sink,
            handoff_sec=0.0,
            trailing_silence_sec=0.05,
            no_speech_sec=1.0,
            hard_cap_sec=1.0,
            poll_sec=0.01,
        )
        assert not timed_out

    asyncio.run(run())


def test_wait_until_speech_done_empty_response() -> None:
    async def run() -> None:
        sink = TraceSink(24000, 480)
        sink.started_at = time.monotonic()
        timed_out = await wait_until_speech_done(
            sink,
            handoff_sec=0.0,
            trailing_silence_sec=1.0,
            no_speech_sec=0.05,
            hard_cap_sec=1.0,
            poll_sec=0.01,
        )
        assert not timed_out

    asyncio.run(run())


def test_wait_until_speech_done_ignores_short_noise() -> None:
    async def run() -> None:
        sink = TraceSink(24000, 480)
        sink.started_at = time.monotonic()
        sink.played_audio.append(PlayedAudio(0.0, 0.02, 0.1))
        timed_out = await wait_until_speech_done(
            sink,
            handoff_sec=0.0,
            trailing_silence_sec=1.0,
            no_speech_sec=0.05,
            hard_cap_sec=1.0,
            poll_sec=0.01,
        )
        assert not timed_out

    asyncio.run(run())


def test_wait_until_speech_done_times_out_when_audio_pending() -> None:
    async def run() -> None:
        sink = TraceSink(24000, 480)
        sink.started_at = time.monotonic()
        await sink.queue.put((0, AudioFrame(np.full(480, 0.1, dtype=np.float32))))
        timed_out = await wait_until_speech_done(
            sink,
            handoff_sec=0.0,
            trailing_silence_sec=1.0,
            no_speech_sec=1.0,
            hard_cap_sec=0.05,
            poll_sec=0.01,
        )
        assert timed_out

    asyncio.run(run())


def test_rollout_eot_waits_for_silence_after_handoff(tmp_path: Path) -> None:
    async def run() -> None:
        config = ModelConfig(
            id="fake",
            name="Fake",
            model_dir=tmp_path,
            adapter_path=tmp_path / "adapter.py",
            connection=ConnectionConfig(url="fake"),
            audio=AudioConfig(
                input_sample_rate=24000,
                output_sample_rate=24000,
                encoding="pcm16",
                frame_samples=480,
            ),
            session=SessionConfig(),
        )
        scenario = EotScenario(
            point_id="point",
            point_dir=tmp_path,
            contexts={ContextType.TEXT: event_context([])},
            input_pcm=np.full(480, 0.05, dtype=np.float32),
            sample_rate=24000,
            checkpoint_sec=0.02,
            window_end_sec=0.02,
            reference=ReferenceResponse(duration_sec=1.0, word_count=4),
        )
        model = FakeModel()
        trace = await run_rollout(
            model,
            config,
            SessionConfig(),
            scenario,
            tmp_path / "result",
            eot_trailing_silence_sec=0.05,
            eot_no_speech_sec=0.05,
            eot_hard_cap_sec=1.0,
        )
        drain = next(event for event in trace.events if event.kind == "eot_drain_done")
        assert drain.data["timed_out"] is False
        assert not trace.timed_out
        assert trace.session["detect_turns"] is False
        assert model.ended_input

    asyncio.run(run())


class SlowResponderModel(FakeModel):
    """First audio arrives only after several frames, like a hosted model whose
    response is still in flight when the fixed pause drain ends."""

    def __init__(self, respond_after_frames: int):
        super().__init__()
        self.respond_after_frames = respond_after_frames

    async def send_audio(self, pcm: np.ndarray) -> None:
        self.sent_frames += 1
        if self.sent_frames == self.respond_after_frames:
            await self.audio.put(AudioFrame(np.full(480, 0.1, dtype=np.float32)))


def test_pause_drain_extends_until_a_slow_response_arrives(tmp_path: Path) -> None:
    """Grok regularly created its response inside the hold but delivered first
    audio after the fixed drain had ended; a live model was then recorded as
    no_model_audio and dropped from the means."""

    async def run() -> None:
        config = ModelConfig(
            id="fake",
            name="Fake",
            model_dir=tmp_path,
            adapter_path=tmp_path / "adapter.py",
            connection=ConnectionConfig(url="fake"),
            audio=AudioConfig(
                input_sample_rate=24000,
                output_sample_rate=24000,
                encoding="pcm16",
                frame_samples=480,
            ),
            session=SessionConfig(),
        )
        scenario = PauseScenario(
            point_id="point",
            point_dir=tmp_path,
            contexts={ContextType.TEXT: event_context([])},
            input_pcm=np.full(480, 0.05, dtype=np.float32),
            pause_input_pcm=np.full(480, 0.05, dtype=np.float32),
            sample_rate=24000,
            checkpoint_sec=0.02,
            window_end_sec=0.02,
        )
        # One live frame plus a one-frame drain end before frame 4; only the
        # bounded extension keeps streaming until the response shows up.
        model = SlowResponderModel(respond_after_frames=4)
        trace = await run_rollout(
            model,
            config,
            SessionConfig(),
            scenario,
            tmp_path / "result",
            pause_drain_sec=0.02,
            pause_no_audio_extension_sec=1.0,
        )

        assert trace.status == "scored"
        assert trace.response_audio.shape[0] > 0
        # The extension stops once audio arrived (plus the onset tail) rather
        # than running to its cap.
        assert model.sent_frames < 4 + 60

    asyncio.run(run())


def test_pause_drain_sec_extends_the_streamed_duplex_rollout() -> None:
    """A duplex pause rollout is only as long as the frames it is streamed, so
    the drain is what decides how long the model runs after the pause."""
    scenario = PauseScenario(
        point_id="point",
        point_dir=Path("."),
        contexts={
            ContextType.DUPLEX_AUDIO: DuplexAudioContext(
                user_pcm=np.zeros(1920, dtype=np.float32),
                assistant_pcm=np.zeros(1920, dtype=np.float32),
                sample_rate=24000,
            )
        },
        input_pcm=np.zeros(1920, dtype=np.float32),
        sample_rate=24000,
        checkpoint_sec=0.08,
        window_end_sec=0.16,
        pause_input_pcm=np.zeros(3840, dtype=np.float32),
    )

    live = scenario.live_pcm(step_driven=True)

    # Live audio is the pause_end turn. Drain silence is streamed after it, on
    # the same frame grid, rather than being concatenated into the input buffer.
    assert live.shape[0] == 3840
    assert silence_frame_count(3.0, 24000, 1920) * 1920 / 24000 == pytest.approx(3.04)
    assert silence_frame_count(15.0, 24000, 1920) * 1920 / 24000 == pytest.approx(15.04)


class DuplexFakeModel(FakeModel):
    @property
    def supported_contexts(self) -> tuple[ContextType, ...]:
        return (ContextType.DUPLEX_AUDIO,)


def test_rollout_eot_keeps_stepping_a_duplex_model_through_the_drain(
    tmp_path: Path,
) -> None:
    """A duplex model runs one LM step per received frame. Waiting out the
    drain on the wall clock would leave it unstepped for the whole response
    being measured, which is the same bug the pause protocol already had."""

    async def run() -> None:
        config = ModelConfig(
            id="fake",
            name="Fake",
            model_dir=tmp_path,
            adapter_path=tmp_path / "adapter.py",
            connection=ConnectionConfig(url="fake"),
            audio=AudioConfig(
                input_sample_rate=24000,
                output_sample_rate=24000,
                encoding="pcm16",
                frame_samples=480,
            ),
            session=SessionConfig(),
        )
        scenario = EotScenario(
            point_id="point",
            point_dir=tmp_path,
            contexts={
                ContextType.DUPLEX_AUDIO: DuplexAudioContext(
                    user_pcm=np.zeros(1920, dtype=np.float32),
                    assistant_pcm=np.zeros(1920, dtype=np.float32),
                    sample_rate=24000,
                )
            },
            input_pcm=np.full(480, 0.05, dtype=np.float32),
            sample_rate=24000,
            checkpoint_sec=0.02,
            window_end_sec=0.02,
            reference=ReferenceResponse(duration_sec=1.0, word_count=4),
        )
        model = DuplexFakeModel()
        trace = await run_rollout(
            model,
            config,
            SessionConfig(),
            scenario,
            tmp_path / "result",
            eot_trailing_silence_sec=0.05,
            eot_no_speech_sec=0.05,
            eot_hard_cap_sec=1.0,
        )

        # One frame of user turn, then real silence frames for the drain.
        assert model.sent_frames > 1
        assert not model.ended_input
        scheduled = [
            event.data["scheduled_at_sec"]
            for event in trace.events
            if event.kind == "input_audio_sent"
        ]
        # The drain frames continue stream_pcm's schedule rather than starting
        # a second clock, so observed_input_time still maps source to wall time.
        assert scheduled == pytest.approx(
            [0.02 * (index + 1) for index in range(len(scheduled))]
        )
        assert len(scheduled) == model.sent_frames

    asyncio.run(run())
