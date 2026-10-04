"""Tests for rollout artifact triage.

Artifacts are produced by the real `run_rollout` against fake models, so the
analyzer is exercised on files with the same shape a real run writes.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np

from bench.audio import MODEL_FRAME_SAMPLES
from bench.points import PauseScenario
from bench.protocol import (
    AudioEvent,
    AudioFrame,
    Context,
    ContextType,
    RealtimeModel,
    SessionConfig,
    SessionMetrics,
    SessionMode,
    event_context,
)
from bench.registry import AudioConfig, ConnectionConfig, ModelConfig
from bench.rollout import run_rollout
from bench.report.analyze import analyze_rollout, analyze_run, format_diagnosis

SAMPLE_RATE = 24000
INPUT_FRAMES = 5


class EchoModel(RealtimeModel):
    """Returns one output frame for every `reply_every` frames it receives."""

    def __init__(self, reply_every: int = 1, amplitude: float = 0.0, say: str = ""):
        self.reply_every = reply_every
        self.amplitude = amplitude
        self.say = say
        self.audio: asyncio.Queue[AudioEvent | None] = asyncio.Queue()
        self.text: asyncio.Queue[str | None] = asyncio.Queue()
        self.received = 0
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
        self.received += 1
        if self.received % self.reply_every:
            return
        await self.audio.put(
            AudioFrame(np.full(MODEL_FRAME_SAMPLES, self.amplitude, dtype=np.float32))
        )
        if self.say:
            await self.text.put(self.say)
            self.say = ""

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


def model_config(tmp_path: Path) -> ModelConfig:
    return ModelConfig(
        id="fake",
        name="Fake",
        model_dir=tmp_path,
        adapter_path=tmp_path / "adapter.py",
        connection=ConnectionConfig(url="fake"),
        audio=AudioConfig(
            input_sample_rate=SAMPLE_RATE,
            output_sample_rate=SAMPLE_RATE,
            encoding="pcm16",
            frame_samples=MODEL_FRAME_SAMPLES,
        ),
        session=SessionConfig(),
    )


def scenario(tmp_path: Path) -> PauseScenario:
    pcm = np.full(INPUT_FRAMES * MODEL_FRAME_SAMPLES, 0.05, dtype=np.float32)
    return PauseScenario(
        point_id="point",
        point_dir=tmp_path,
        contexts={ContextType.TEXT: event_context([])},
        input_pcm=pcm,
        pause_input_pcm=pcm,
        sample_rate=SAMPLE_RATE,
        checkpoint_sec=INPUT_FRAMES * 0.08,
        window_end_sec=INPUT_FRAMES * 0.08 + 0.1,
    )


def write_rollout(tmp_path: Path, model: EchoModel, name: str = "rollout_001") -> Path:
    rollout_dir = tmp_path / name

    async def run() -> None:
        await run_rollout(
            model,
            model_config(tmp_path),
            SessionConfig(),
            scenario(tmp_path),
            rollout_dir,
            pause_drain_sec=0.05,
            pause_no_audio_extension_sec=0.05,
        )

    asyncio.run(run())
    return rollout_dir


def test_healthy_rollout_is_not_flagged(tmp_path: Path) -> None:
    rollout_dir = write_rollout(tmp_path, EchoModel())
    diagnosis = analyze_rollout(rollout_dir)
    # The pause_end frames plus one frame of streamed drain silence.
    assert diagnosis.input_frames == INPUT_FRAMES + 1
    assert diagnosis.output_input_ratio > 0.9
    assert diagnosis.max_input_lag_sec < 0.08
    assert diagnosis.findings == []
    assert diagnosis.ok


def test_dropped_frames_show_up_as_a_low_output_ratio(tmp_path: Path) -> None:
    rollout_dir = write_rollout(tmp_path, EchoModel(reply_every=3))
    diagnosis = analyze_rollout(rollout_dir)
    assert diagnosis.output_input_ratio < 0.5
    assert any("frames were dropped" in finding for finding in diagnosis.findings)
    assert not diagnosis.ok


def test_a_sustained_drone_is_flagged(tmp_path: Path) -> None:
    rollout_dir = write_rollout(tmp_path, EchoModel(amplitude=0.3))
    diagnosis = analyze_rollout(rollout_dir)
    assert diagnosis.response.speech_ratio > 0.5
    assert any("drone" in finding for finding in diagnosis.findings)


def test_audio_above_the_gate_with_no_model_text_is_flagged(tmp_path: Path) -> None:
    """The shape of the wordless phonation in the PersonaPlex pause rollouts:
    plenty of audio over the speech gate and an empty text stream."""
    rollout_dir = write_rollout(tmp_path, EchoModel(amplitude=0.3))
    diagnosis = analyze_rollout(rollout_dir)
    assert diagnosis.response.speech_ratio > 0.05
    assert any("emitted no text" in finding for finding in diagnosis.findings)


def test_the_same_audio_with_model_text_is_not_flagged_as_wordless(
    tmp_path: Path,
) -> None:
    rollout_dir = write_rollout(tmp_path, EchoModel(amplitude=0.3, say="hello"))
    diagnosis = analyze_rollout(rollout_dir)
    assert diagnosis.response.speech_ratio > 0.05
    assert not any("emitted no text" in finding for finding in diagnosis.findings)


def test_a_stale_score_file_is_flagged(tmp_path: Path) -> None:
    rollout_dir = write_rollout(tmp_path, EchoModel())
    with (rollout_dir / "score.json").open("w", encoding="utf-8") as handle:
        json.dump({"score": 0}, handle)
    diagnosis = analyze_rollout(rollout_dir)
    assert diagnosis.recorded_score == 0
    assert diagnosis.recomputed_score == 1
    assert any("score.json" in finding for finding in diagnosis.findings)


def test_a_rollout_that_returned_nothing_is_flagged_not_passed(tmp_path: Path) -> None:
    """The exact shape of a structural false pass: full input trace, no output,
    and a score file that would once have read 1."""
    rollout_dir = write_rollout(tmp_path, EchoModel(reply_every=10**6))
    diagnosis = analyze_rollout(rollout_dir)

    assert diagnosis.status == "no_model_audio"
    assert diagnosis.recomputed_score is None
    assert any("nothing to score" in finding for finding in diagnosis.findings)
    assert not diagnosis.ok


def test_analyze_run_walks_every_rollout(tmp_path: Path) -> None:
    write_rollout(tmp_path / "point", EchoModel(), name="rollout_001")
    write_rollout(tmp_path / "point", EchoModel(reply_every=3), name="rollout_002")
    diagnoses = analyze_run(tmp_path)
    assert len(diagnoses) == 2
    assert [diagnosis.ok for diagnosis in diagnoses] == [True, False]
    assert "FLAG" in format_diagnosis(diagnoses[1])
