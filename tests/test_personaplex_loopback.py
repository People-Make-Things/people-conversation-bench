"""End-to-end PersonaPlex pipeline over loopback, with no GPU and no weights.

These drive the production client adapter against the real server coroutines
through real websockets and the real Opus codec. What they can prove is that the
transport conserves audio and that the session lifecycle behaves; what they
cannot prove is anything about the 7B model itself.
"""

from __future__ import annotations

import asyncio

import live_session
import numpy as np
import personaplex_harness as harness
import prefill
import pytest
import websockets
from aiohttp import web

from bench.audio import MODEL_FRAME_SAMPLES, TARGET_SAMPLE_RATE
from bench.audio_metrics import best_lag_correlation, discontinuity_ratio, rms
from bench.points import PauseScenario
from bench.protocol import (
    ContextType,
    DuplexAudioContext,
    SessionConfig,
    SessionMode,
    TimedWord,
)
from bench.rollout import run_rollout
from models.personaplex.adapter import PersonaPlexModel

# Opus adds ~6.5 ms of algorithmic delay, so the tail of the stream can still be
# in flight when a test stops streaming.
SETTLE_SEC = 0.4
FRAME_COUNT = 16


async def collect_output(model: PersonaPlexModel) -> tuple[asyncio.Task, list]:
    frames: list[np.ndarray] = []

    async def pump() -> None:
        async for event in model.recv_audio():
            frames.append(event.pcm)

    return asyncio.create_task(pump()), frames


async def stream(model: PersonaPlexModel, pcm: np.ndarray, chunk: int, gap_sec: float) -> None:
    for offset in range(0, pcm.shape[0], chunk):
        await model.send_audio(pcm[offset : offset + chunk])
        await asyncio.sleep(gap_sec)


@pytest.mark.parametrize("chunk", [480, 960, 1920, 2880])
def test_server_consumes_every_sample_for_any_client_chunk_size(chunk: int) -> None:
    """Opus only accepts a fixed set of append sizes; 480/960/2880 are all
    misaligned with the 1920-sample model frame, which is exactly the case the
    pre-fix server discarded."""

    async def run() -> None:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(voice_prompt="NATF2.pt"))
            signal = harness.frame_ladder(FRAME_COUNT)
            await stream(model, signal, chunk, 0.006)
            await asyncio.sleep(SETTLE_SEC)
            received = state.mimi.received_pcm()
            await model.close()

        assert received.shape[0] % MODEL_FRAME_SAMPLES == 0
        assert received.shape[0] >= signal.shape[0] - MODEL_FRAME_SAMPLES
        assert state.lm_gen.live_steps == received.shape[0] // MODEL_FRAME_SAMPLES

    asyncio.run(run())


def test_server_receives_the_waveform_the_client_sent() -> None:
    async def run() -> None:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(voice_prompt="NATF2.pt"))
            signal = harness.frame_ladder(FRAME_COUNT)
            await stream(model, signal, MODEL_FRAME_SAMPLES, 0.01)
            await asyncio.sleep(SETTLE_SEC)
            received = state.mimi.received_pcm()
            await model.close()

        length = min(signal.shape[0], received.shape[0])
        lag, correlation = best_lag_correlation(signal[:length], received[:length])
        assert correlation > 0.9
        assert abs(lag) < MODEL_FRAME_SAMPLES
        # A dropped chunk splices the waveform and leaves a step edge; a clean
        # stream of bin-aligned tones stays smooth.
        assert discontinuity_ratio(received) < 0.2

    asyncio.run(run())


def test_no_audio_is_lost_while_the_model_blocks_the_event_loop() -> None:
    async def run() -> None:
        state = harness.FakeServerState(lm_gen=harness.FakeLMGen(step_delay_sec=0.03))
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(voice_prompt="NATF2.pt"))
            signal = harness.frame_ladder(FRAME_COUNT)
            await stream(model, signal, 480, 0.004)
            await asyncio.sleep(1.5)
            received = state.mimi.received_pcm()
            await model.close()

        assert received.shape[0] >= signal.shape[0] - MODEL_FRAME_SAMPLES

    asyncio.run(run())


def test_model_clock_tracks_wall_clock_when_fed_at_realtime() -> None:
    """The model steps once per received frame, so dropped frames make its
    internal clock run slow and pause timing is then measured against a
    drifting reference."""
    realtime_frames = 8

    async def run() -> tuple[int, float]:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(voice_prompt="NATF2.pt"))
            started = asyncio.get_running_loop().time()
            await stream(
                model,
                harness.frame_ladder(realtime_frames),
                MODEL_FRAME_SAMPLES,
                MODEL_FRAME_SAMPLES / TARGET_SAMPLE_RATE,
            )
            await asyncio.sleep(SETTLE_SEC)
            elapsed = asyncio.get_running_loop().time() - started - SETTLE_SEC
            steps = state.lm_gen.live_steps
            await model.close()
        return steps, elapsed

    steps, elapsed = asyncio.run(run())
    model_clock_sec = steps * MODEL_FRAME_SAMPLES / TARGET_SAMPLE_RATE
    assert abs(model_clock_sec - elapsed) < 2 * MODEL_FRAME_SAMPLES / TARGET_SAMPLE_RATE


def test_model_output_returns_to_the_client_with_text() -> None:
    async def run() -> tuple[int, list[str]]:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(voice_prompt="NATF2.pt"))
            audio_task, frames = await collect_output(model)
            tokens: list[str] = []

            async def pump_text() -> None:
                async for token in model.recv_text():
                    tokens.append(token)

            text_task = asyncio.create_task(pump_text())
            await stream(model, harness.frame_ladder(FRAME_COUNT), MODEL_FRAME_SAMPLES, 0.01)
            await asyncio.sleep(SETTLE_SEC)
            await model.close()
            await asyncio.sleep(0.05)
            audio_task.cancel()
            text_task.cancel()
        return sum(frame.shape[0] for frame in frames), tokens

    samples, tokens = asyncio.run(run())
    assert samples % MODEL_FRAME_SAMPLES == 0
    assert samples >= (FRAME_COUNT - 2) * MODEL_FRAME_SAMPLES
    assert set(tokens) == {" ok"}
    assert len(tokens) >= FRAME_COUNT // harness.TEXT_EVERY_N_FRAMES - 1


def test_session_config_reaches_the_server() -> None:
    async def run() -> harness.FakeServerState:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(
                SessionConfig(voice_prompt="NATF2.pt", text_prompt="be brief", seed=99)
            )
            await asyncio.sleep(0.05)
            await model.close()
        return state

    state = asyncio.run(run())
    assert state.seeds == [99]
    assert state.voice_prompts == ["NATF2.pt"]
    assert state.text_prompts == ["be brief"]
    assert state.lm_gen.system_prompt_calls == 1
    assert state.lm_gen.resets == 1


def test_empty_voice_prompt_clears_the_previous_session_voice() -> None:
    """Eval sends no voice prompt so the teacher-forced history sets the voice.

    That makes clearing load-bearing: the process is reused across rollouts, so
    without it every later rollout keeps the first one's voice.
    """

    async def run() -> harness.FakeServerState:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            for voice_prompt in ("NATF2.pt", ""):
                model = PersonaPlexModel(harness.model_config(url))
                await model.connect(SessionConfig(voice_prompt=voice_prompt))
                await asyncio.sleep(0.05)
                await model.close()
        return state

    state = asyncio.run(run())
    assert state.voice_prompts == ["NATF2.pt", None]
    assert state.lm_gen.voice_prompt_path is None


def test_preload_url_carries_no_session_fields() -> None:
    """A persona prompt in the query string can exceed the request line limit."""
    session = SessionConfig(text_prompt="x" * 9000, voice_prompt="NATF2.pt", seed=1)

    async def run() -> str:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(session, mode=SessionMode.PRELOAD)
            connected = model._ws.request.path
            await model.close()
        return connected

    assert asyncio.run(run()) == "/api/duplex"


def test_idle_session_releases_the_model_lock(monkeypatch) -> None:
    """A peer that vanishes without a close frame used to hold the global lock
    until the container timed out, blocking every later rollout."""
    monkeypatch.setattr(live_session, "SESSION_IDLE_TIMEOUT_SEC", 0.3)

    async def run() -> bool:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig())
            await asyncio.sleep(1.0)
            free = not state.lock.locked()
            await model.close()
        return free

    assert asyncio.run(run())


def test_duplex_preload_primes_history_then_streams_live() -> None:
    history_frames = 8

    async def run() -> harness.FakeServerState:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            session = SessionConfig(voice_prompt="NATF2.pt", text_prompt="hello", seed=5)
            await model.connect(session, mode=SessionMode.PRELOAD)
            await model.preload_context(
                DuplexAudioContext(
                    user_pcm=harness.frame_ladder(history_frames),
                    assistant_pcm=harness.frame_ladder(history_frames) * 0.5,
                    sample_rate=TARGET_SAMPLE_RATE,
                    words=(TimedWord(text="hi", start_sec=0.16, end_sec=0.4),),
                )
            )
            await stream(model, harness.frame_ladder(4), MODEL_FRAME_SAMPLES, 0.01)
            await asyncio.sleep(SETTLE_SEC)
            await model.close()
        return state

    state = asyncio.run(run())
    assert state.lm_gen.prefill_steps == history_frames
    assert len(state.lm_gen.prefill_text_tokens) == history_frames
    assert state.other_mimi.encoded and len(state.other_mimi.encoded) == history_frames
    assert state.lm_gen.live_steps >= 3
    assert state.seeds == [5]


def preload_forced_text(words: tuple[TimedWord, ...], history_frames: int) -> list[int]:
    async def run() -> harness.FakeServerState:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(), mode=SessionMode.PRELOAD)
            await model.preload_context(
                DuplexAudioContext(
                    user_pcm=harness.frame_ladder(history_frames),
                    assistant_pcm=harness.frame_ladder(history_frames) * 0.5,
                    sample_rate=TARGET_SAMPLE_RATE,
                    words=words,
                )
            )
            await model.close()
        return state

    return asyncio.run(run()).lm_gen.prefill_text_tokens


def test_transcript_words_reach_the_forced_text_stream() -> None:
    """The count of forced tokens matching the frame count says nothing about
    whether any word survived JSON, the wire, and alignment. A history whose
    forced stream is entirely PAD is indistinguishable from a working one by
    frame count alone, and that is the shape of a model that never speaks."""
    words = (
        TimedWord(text="hi", start_sec=0.16, end_sec=0.4),
        TimedWord(text="yes", start_sec=0.48, end_sec=0.8),
    )
    forced = preload_forced_text(words, history_frames=10)

    assert forced == [
        harness.TEXT_PAD,
        harness.TEXT_EPAD,
        ord("h"),
        ord("i"),
        harness.TEXT_PAD,
        harness.TEXT_EPAD,
        ord("y"),
        ord("e"),
        ord("s"),
        harness.TEXT_PAD,
    ]
    assert preload_forced_text((), 10) == [harness.TEXT_PAD] * 10


def pause_scenario(tmp_path, checkpoint_frames: int, window_frames: int) -> PauseScenario:
    frame_sec = MODEL_FRAME_SAMPLES / TARGET_SAMPLE_RATE
    return PauseScenario(
        point_id="point",
        point_dir=tmp_path,
        contexts={
            ContextType.DUPLEX_AUDIO: DuplexAudioContext(
                user_pcm=harness.frame_ladder(2),
                assistant_pcm=harness.frame_ladder(2) * 0.5,
                sample_rate=TARGET_SAMPLE_RATE,
                words=(),
            )
        },
        input_pcm=harness.frame_ladder(checkpoint_frames),
        sample_rate=TARGET_SAMPLE_RATE,
        checkpoint_sec=checkpoint_frames * frame_sec,
        window_end_sec=window_frames * frame_sec,
        pause_input_pcm=harness.frame_ladder(window_frames),
    )


def test_duplex_model_is_stepped_through_the_scored_pause_window(tmp_path) -> None:
    """A duplex model only generates on frames it receives.

    Waiting out the pause on the wall clock meant it was never stepped inside
    the window being scored, so it could not have barged in even if it wanted
    to and every rollout drifted toward a pass for the wrong reason.
    """
    checkpoint_frames, window_frames, drain_frames = 4, 12, 2

    async def run() -> tuple[harness.FakeServerState, list[str]]:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            config = harness.model_config(url)
            model = PersonaPlexModel(config)
            trace = await run_rollout(
                model,
                config,
                SessionConfig(text_prompt="be brief"),
                pause_scenario(tmp_path, checkpoint_frames, window_frames),
                tmp_path / "result",
                pause_drain_sec=drain_frames * MODEL_FRAME_SAMPLES / TARGET_SAMPLE_RATE,
            )
        return state, [event.kind for event in trace.events]

    state, kinds = asyncio.run(run())
    # Server prefill diagnostics reach the trace through the existing signal
    # path, which is where history truncation shows up on a real run.
    assert "duplex_primed" in kinds
    steps = state.lm_gen.live_steps
    # Frames are streamed in order, so step N is the model at frame N of the
    # turn. Steps landing between the checkpoint and the window end are exactly
    # the ones it needs to be able to speak during the pause.
    steps_in_window = min(steps, window_frames) - checkpoint_frames
    assert steps_in_window >= window_frames - checkpoint_frames - 1
    assert steps >= window_frames + drain_frames - 1


def test_server_stays_responsive_while_prefilling_a_long_history() -> None:
    """Prefill is one blocking LM step per history frame.

    Run synchronously it pins the event loop for the whole history, so the
    server cannot answer pings and the client tears the session down mid-prefill
    with a 1011. A second connection completing is the cheap proof the loop is
    still turning.
    """
    history_frames = 200
    prefill_delay_sec = 0.005
    prefill_sec = history_frames * prefill_delay_sec

    async def run() -> tuple[float, bool]:
        state = harness.FakeServerState(
            lm_gen=harness.FakeLMGen(prefill_delay_sec=prefill_delay_sec)
        )
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(), mode=SessionMode.PRELOAD)
            preload = asyncio.create_task(
                model.preload_context(
                    DuplexAudioContext(
                        user_pcm=harness.frame_ladder(history_frames),
                        assistant_pcm=harness.frame_ladder(history_frames) * 0.5,
                        sample_rate=TARGET_SAMPLE_RATE,
                        words=(),
                    )
                )
            )
            await asyncio.sleep(prefill_sec / 4)
            started = asyncio.get_running_loop().time()
            probe = PersonaPlexModel(harness.model_config(url))
            await asyncio.wait_for(
                probe.connect(SessionConfig(), mode=SessionMode.PRELOAD),
                timeout=prefill_sec,
            )
            latency = asyncio.get_running_loop().time() - started
            still_prefilling = not preload.done()
            await probe.close()
            await preload
            await model.close()
        return latency, still_prefilling

    latency, still_prefilling = asyncio.run(run())
    assert still_prefilling
    # A handshake needs several loop turns and each one costs a frame, so this
    # is bounded by step time, not by the length of the history.
    assert latency < prefill_sec / 4


SILENT_PREFILL_FRAMES = 100
# aiohttp closes a heartbeated socket once a pong is overdue by half the
# interval, so 1.5x the interval is the deadline an unread prefill has to beat.
HEARTBEAT_CLOSE_FACTOR = 1.5
UNREAD_PREFILL_SEC = 0.5


def silent_prefill_delay_sec(heartbeat: float | None) -> float:
    """Per-frame prefill cost that leaves the socket unread past `heartbeat`.

    Scaling the cost rather than the history keeps this honest for any interval:
    a history long enough to outlast a 20 s heartbeat would be truncated back
    under the deadline by the context window.
    """
    unread_sec = (
        2 * HEARTBEAT_CLOSE_FACTOR * heartbeat if heartbeat else UNREAD_PREFILL_SEC
    )
    return unread_sec / SILENT_PREFILL_FRAMES


async def preload_history_with_a_silent_client(state: harness.FakeServerState) -> None:
    """Prefill a history; the client sends nothing between commit and primed."""
    async with harness.serve(state) as url:
        model = PersonaPlexModel(harness.model_config(url))
        await model.connect(SessionConfig(), mode=SessionMode.PRELOAD)
        await model.preload_context(
            DuplexAudioContext(
                user_pcm=harness.frame_ladder(SILENT_PREFILL_FRAMES),
                assistant_pcm=harness.frame_ladder(SILENT_PREFILL_FRAMES) * 0.5,
                sample_rate=TARGET_SAMPLE_RATE,
                words=(),
            )
        )
        await model.close()


def test_prefill_survives_with_nothing_reading_the_socket() -> None:
    """The session routes must not set a websocket heartbeat.

    Nothing reads the socket between the duplex commit and the primed message,
    and aiohttp only observes a pong from inside `ws.receive()`, so a heartbeat
    closes the connection once its pong is overdue however promptly the client
    answered. On the live server that failed every point whose history ran past
    the deadline and passed every shorter one.
    """
    delay = silent_prefill_delay_sec(None)
    state = harness.FakeServerState(lm_gen=harness.FakeLMGen(prefill_delay_sec=delay))
    asyncio.run(preload_history_with_a_silent_client(state))
    assert state.lm_gen.prefill_steps == SILENT_PREFILL_FRAMES


def test_a_heartbeat_closes_the_socket_during_prefill(monkeypatch) -> None:
    """Why the test above is not vacuous: with a heartbeat set, the same silent
    prefill never reaches the primed message."""

    async def open_heartbeated_socket(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse(
            max_msg_size=live_session.MAX_WS_MSG_SIZE, heartbeat=0.2
        )
        await ws.prepare(request)
        return ws

    monkeypatch.setattr(harness, "open_session_socket", open_heartbeated_socket)
    delay = silent_prefill_delay_sec(0.2)
    state = harness.FakeServerState(lm_gen=harness.FakeLMGen(prefill_delay_sec=delay))
    with pytest.raises(websockets.ConnectionClosed):
        asyncio.run(preload_history_with_a_silent_client(state))


def test_prefill_longer_than_the_idle_timeout_still_streams(monkeypatch) -> None:
    """The idle watchdog measures how long the peer has been quiet.

    Prefill of a long history can outlast the timeout on its own, and the
    session would then be closed the moment it went live.
    """
    monkeypatch.setattr(
        live_session, "SESSION_IDLE_TIMEOUT_SEC", UNREAD_PREFILL_SEC * 0.8
    )
    delay = silent_prefill_delay_sec(None)

    async def run() -> harness.FakeServerState:
        state = harness.FakeServerState(lm_gen=harness.FakeLMGen(prefill_delay_sec=delay))
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(), mode=SessionMode.PRELOAD)
            await model.preload_context(
                DuplexAudioContext(
                    user_pcm=harness.frame_ladder(SILENT_PREFILL_FRAMES),
                    assistant_pcm=harness.frame_ladder(SILENT_PREFILL_FRAMES) * 0.5,
                    sample_rate=TARGET_SAMPLE_RATE,
                    words=(),
                )
            )
            await stream(model, harness.frame_ladder(4), MODEL_FRAME_SAMPLES, 0.01)
            await asyncio.sleep(SETTLE_SEC)
            await model.close()
        return state

    state = asyncio.run(run())
    assert state.lm_gen.prefill_steps == SILENT_PREFILL_FRAMES
    assert state.lm_gen.live_steps >= 3


def test_history_longer_than_the_context_window_is_truncated_from_the_front(
    monkeypatch,
) -> None:
    """Past the model's context window the ring would silently evict the
    identity frame and the start of the conversation instead of the oldest audio."""
    monkeypatch.setattr(prefill, "MAX_HISTORY_FRAMES", 6)
    history_frames = 10

    async def run() -> harness.FakeServerState:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(), mode=SessionMode.PRELOAD)
            await model.preload_context(
                DuplexAudioContext(
                    user_pcm=harness.frame_ladder(history_frames),
                    assistant_pcm=harness.frame_ladder(history_frames) * 0.5,
                    sample_rate=TARGET_SAMPLE_RATE,
                    words=(),
                )
            )
            await model.close()
        return state

    state = asyncio.run(run())
    assert state.lm_gen.prefill_steps == 6
    # The tail is what survives: the most recent audio, not the oldest.
    kept = np.concatenate(state.mimi.encoded)
    expected = harness.frame_ladder(history_frames)[-kept.shape[0] :]
    assert np.allclose(kept, expected, atol=1.0 / 32768.0)


def test_silence_in_stays_silent_out() -> None:
    async def run() -> np.ndarray:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = PersonaPlexModel(harness.model_config(url))
            await model.connect(SessionConfig(voice_prompt="NATF2.pt"))
            audio_task, frames = await collect_output(model)
            silence = np.zeros(FRAME_COUNT * MODEL_FRAME_SAMPLES, dtype=np.float32)
            await stream(model, silence, MODEL_FRAME_SAMPLES, 0.01)
            await asyncio.sleep(SETTLE_SEC)
            await model.close()
            await asyncio.sleep(0.05)
            audio_task.cancel()
        return np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)

    output = asyncio.run(run())
    assert output.shape[0] > 0
    assert rms(output) < 1e-3
