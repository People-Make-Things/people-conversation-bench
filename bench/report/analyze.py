"""Flag suspect rollouts from their written artifacts.

Reads `trace.json`, `score.json`, and `response.wav` out of a rollout directory
and reports what can be checked without listening: whether the model consumed
every frame the harness streamed, whether the harness itself kept pace, and
whether the returned audio looks like speech.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.audio import MODEL_FRAME_SAMPLES, read_wav_segment
from bench.audio_metrics import AudioReport, describe
from bench.metrics.pause import MAX_WORDLESS_SPEECH_RATIO, score_pause_recognition
from bench.trace import NO_MODEL_AUDIO, RolloutTrace

# A duplex model steps once per received user frame and emits one frame back, so
# output well under input means frames were dropped or the server fell behind.
MIN_OUTPUT_INPUT_RATIO = 0.9
# The harness sends one frame per frame duration; more lag than that breaks the
# source-time to wall-clock mapping pause scoring depends on.
MAX_INPUT_LAG_SEC = MODEL_FRAME_SAMPLES / 24000
MAX_CLIPPING_RATIO = 0.01
# Heuristic: speech keeps changing spectrally, a sustained non-speech
# vocalization does not. Calibrate against a known-good rollout before trusting.
MIN_SPEECH_SPECTRAL_FLUX = 0.05
DRONE_SPEECH_RATIO = 0.5


@dataclass(frozen=True)
class RolloutDiagnosis:
    rollout_dir: Path
    point_id: str
    model_id: str
    input_frames: int
    input_streamed_sec: float
    model_audio_sec: float
    output_input_ratio: float
    max_input_lag_sec: float
    response: AudioReport
    recorded_score: int | None
    recomputed_score: int | None
    status: str
    findings: list[str]

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict:
        return {
            "rollout_dir": str(self.rollout_dir),
            "point_id": self.point_id,
            "model_id": self.model_id,
            "input_frames": self.input_frames,
            "input_streamed_sec": self.input_streamed_sec,
            "model_audio_sec": self.model_audio_sec,
            "output_input_ratio": self.output_input_ratio,
            "max_input_lag_sec": self.max_input_lag_sec,
            "response": self.response.to_dict(),
            "recorded_score": self.recorded_score,
            "recomputed_score": self.recomputed_score,
            "status": self.status,
            "findings": self.findings,
        }


def load_trace(rollout_dir: Path) -> RolloutTrace:
    with (rollout_dir / "trace.json").open(encoding="utf-8") as handle:
        return RolloutTrace.from_dict(json.load(handle))


def recorded_score(rollout_dir: Path) -> int | None:
    score_path = rollout_dir / "score.json"
    if not score_path.is_file():
        return None
    with score_path.open(encoding="utf-8") as handle:
        return json.load(handle).get("score")


def input_timing(trace: RolloutTrace) -> tuple[int, float, float]:
    sent = [event for event in trace.events if event.kind == "input_audio_sent"]
    streamed = sum(event.duration_sec or 0.0 for event in sent)
    lag = max(
        (
            event.at_sec - float(event.data.get("scheduled_at_sec", event.at_sec))
            for event in sent
        ),
        default=0.0,
    )
    return len(sent), streamed, lag


def model_audio_duration(trace: RolloutTrace) -> float:
    return sum(
        event.duration_sec or 0.0
        for event in trace.events
        if event.kind == "model_audio_received"
    )


def collect_findings(
    streamed_sec: float,
    ratio: float,
    lag: float,
    response: AudioReport,
    recorded: int | None,
    recomputed: int | None,
    status: str,
    text: str,
) -> list[str]:
    findings: list[str] = []
    if status == NO_MODEL_AUDIO:
        findings.append(
            "no model audio was received, so there is nothing to score; "
            "the model was never stepped or the transport died"
        )
    if streamed_sec > 0 and ratio < MIN_OUTPUT_INPUT_RATIO:
        findings.append(
            f"model returned {ratio:.2f}x the streamed input duration; "
            "frames were dropped or the server ran behind wall clock"
        )
    if lag > MAX_INPUT_LAG_SEC:
        findings.append(
            f"input streaming fell {lag * 1000:.0f} ms behind schedule; "
            "pause timing is measured against a drifting reference"
        )
    if response.clipping_ratio > MAX_CLIPPING_RATIO:
        findings.append(f"{response.clipping_ratio:.1%} of response samples are clipped")
    if (
        response.speech_ratio > DRONE_SPEECH_RATIO
        and response.spectral_flux < MIN_SPEECH_SPECTRAL_FLUX
    ):
        findings.append(
            f"response is loud but spectrally static (flux {response.spectral_flux:.3f}); "
            "likely a drone rather than speech"
        )
    if not text.strip() and response.speech_ratio > MAX_WORDLESS_SPEECH_RATIO:
        findings.append(
            f"{response.speech_ratio:.0%} of the response is above the speech gate but "
            "the model emitted no text; wordless phonation, not speech"
        )
    if recorded is not None and recorded != recomputed:
        findings.append(f"score.json says {recorded} but the trace scores {recomputed}")
    return findings


def analyze_rollout(rollout_dir: Path) -> RolloutDiagnosis:
    trace = load_trace(rollout_dir)
    frames, streamed_sec, lag = input_timing(trace)
    model_sec = model_audio_duration(trace)
    ratio = model_sec / streamed_sec if streamed_sec > 0 else 0.0

    response_path = rollout_dir / "response.wav"
    pcm = (
        read_wav_segment(response_path, target_rate=trace.sample_rate)[0]
        if response_path.is_file()
        else np.zeros(0, dtype=np.float32)
    )
    response = describe(pcm, trace.sample_rate)

    recorded = recorded_score(rollout_dir)
    # Only pause rollouts carry a single score to cross-check; response-length
    # records ratios instead, and its score.json has no "score" key.
    recomputed = (
        score_pause_recognition(trace).score
        if trace.expected_action == "pause"
        else None
    )
    return RolloutDiagnosis(
        rollout_dir=rollout_dir,
        point_id=trace.point_id,
        model_id=trace.model_id,
        input_frames=frames,
        input_streamed_sec=streamed_sec,
        model_audio_sec=model_sec,
        output_input_ratio=ratio,
        max_input_lag_sec=lag,
        response=response,
        recorded_score=recorded,
        recomputed_score=recomputed,
        status=trace.status,
        findings=collect_findings(
            streamed_sec,
            ratio,
            lag,
            response,
            recorded,
            recomputed,
            trace.status,
            trace.text,
        ),
    )


def analyze_run(run_dir: Path) -> list[RolloutDiagnosis]:
    return [
        analyze_rollout(trace_path.parent)
        for trace_path in sorted(run_dir.glob("**/trace.json"))
    ]


def format_diagnosis(diagnosis: RolloutDiagnosis) -> str:
    header = (
        f"{'ok  ' if diagnosis.ok else 'FLAG'} {diagnosis.rollout_dir} "
        f"score={diagnosis.recomputed_score} "
        f"out/in={diagnosis.output_input_ratio:.2f} "
        f"lag={diagnosis.max_input_lag_sec * 1000:.0f}ms "
        f"flux={diagnosis.response.spectral_flux:.3f}"
    )
    return "\n".join([header] + [f"       - {finding}" for finding in diagnosis.findings])


def run(run_dir: str | Path) -> None:
    diagnoses = analyze_run(Path(run_dir))
    if not diagnoses:
        print(f"No rollouts found under {run_dir}")
        return
    for diagnosis in diagnoses:
        print(format_diagnosis(diagnosis))
    flagged = [diagnosis for diagnosis in diagnoses if not diagnosis.ok]
    print(f"\n{len(flagged)} of {len(diagnoses)} rollout(s) flagged")
