"""Moshi uses the shared duplex loopback: audio is conserved and the model is stepped."""

from __future__ import annotations

import asyncio
import inspect

import live_session
import personaplex_harness as harness
from test_personaplex_loopback import (
    FRAME_COUNT,
    SETTLE_SEC,
    pause_scenario,
    stream,
)

from bench.audio import MODEL_FRAME_SAMPLES, TARGET_SAMPLE_RATE
from bench.protocol import SessionConfig
from bench.registry import ConnectionConfig, ModelConfig, load_adapter, load_model_config, resolve_model_ref
from bench.rollout import run_rollout


def moshi_loopback_config(base_url: str) -> ModelConfig:
    config = load_model_config(resolve_model_ref("moshi"))
    return ModelConfig(
        id=config.id,
        name=config.name,
        model_dir=config.model_dir,
        adapter_path=config.adapter_path,
        connection=ConnectionConfig(url=base_url),
        audio=config.audio,
        session=config.session,
        max_concurrency=config.max_concurrency,
    )


def test_moshi_server_consumes_every_sample() -> None:
    async def run() -> None:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            model = load_adapter(moshi_loopback_config(url))
            await model.connect(SessionConfig())
            signal = harness.frame_ladder(FRAME_COUNT)
            await stream(model, signal, 1920, 0.006)
            await asyncio.sleep(SETTLE_SEC)
            received = state.mimi.received_pcm()
            await model.close()

        assert received.shape[0] % MODEL_FRAME_SAMPLES == 0
        assert received.shape[0] >= signal.shape[0] - MODEL_FRAME_SAMPLES
        assert state.lm_gen.live_steps == received.shape[0] // MODEL_FRAME_SAMPLES

    asyncio.run(run())


def test_moshi_is_stepped_through_the_scored_pause_window(tmp_path) -> None:
    checkpoint_frames, window_frames, drain_frames = 4, 12, 2

    async def run() -> tuple[harness.FakeServerState, list[str]]:
        state = harness.FakeServerState()
        async with harness.serve(state) as url:
            config = moshi_loopback_config(url)
            model = load_adapter(config)
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
    assert "duplex_primed" in kinds
    steps = state.lm_gen.live_steps
    steps_in_window = min(steps, window_frames) - checkpoint_frames
    assert steps_in_window >= window_frames - checkpoint_frames - 1
    assert steps >= window_frames + drain_frames - 1


def test_moshi_routes_open_sockets_without_a_heartbeat() -> None:
    source = inspect.getsource(live_session.open_session_socket)
    assert "heartbeat=None" in source
    harness_source = inspect.getsource(harness.serve)
    assert harness_source.count("open_session_socket") >= 2
