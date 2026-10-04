"""Score-file vocabulary shared by the runner, scoring, and report."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bench.speech import PlayedAudio

SCORED = "scored"
NO_MODEL_AUDIO = "no_model_audio"


@dataclass(frozen=True)
class TraceEvent:
    kind: str
    at_sec: float
    duration_sec: float | None = None
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {"kind": self.kind, "at_sec": self.at_sec}
        if self.duration_sec is not None:
            result["duration_sec"] = self.duration_sec
        if self.data:
            result["data"] = self.data
        return result

    @classmethod
    def from_dict(cls, data: dict) -> TraceEvent:
        return cls(
            kind=data["kind"],
            at_sec=float(data["at_sec"]),
            duration_sec=data.get("duration_sec"),
            data=data.get("data", {}),
        )


@dataclass
class RolloutTrace:
    point_id: str
    model_id: str
    rollout_index: int
    expected_action: str
    checkpoint_sec: float
    window_end_sec: float
    events: list[TraceEvent]
    played_audio: list[PlayedAudio]
    text: str
    response_audio: np.ndarray
    sample_rate: int
    session: dict = field(default_factory=dict)
    timed_out: bool = False
    played_pcm: list[np.ndarray] = field(default_factory=list)

    @property
    def model_frames_received(self) -> int:
        return sum(1 for event in self.events if event.kind == "model_audio_received")

    @property
    def model_audio_sec(self) -> float:
        return self.response_audio.shape[0] / self.sample_rate

    @property
    def status(self) -> str:
        # Silence is only evidence of anything if the model was alive and
        # generating it. With no audio at all there is nothing to score.
        return SCORED if self.played_audio else NO_MODEL_AUDIO

    def to_dict(self) -> dict:
        return {
            "point_id": self.point_id,
            "model_id": self.model_id,
            "rollout_index": self.rollout_index,
            "expected_action": self.expected_action,
            "checkpoint_sec": self.checkpoint_sec,
            "window_end_sec": self.window_end_sec,
            "timed_out": self.timed_out,
            "events": [event.to_dict() for event in self.events],
            "played_audio": [
                {
                    "start_sec": audio.start_sec,
                    "end_sec": audio.end_sec,
                    "rms": audio.rms,
                }
                for audio in self.played_audio
            ],
            "text": self.text,
            "model_frames_received": self.model_frames_received,
            "model_audio_sec": self.model_audio_sec,
            "response_wav": "response.wav",
            "rollout_wav": "rollout.wav",
            "rollout_channels": {"left": "user", "right": "model"},
            "session": self.session,
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> RolloutTrace:
        return cls(
            point_id=payload["point_id"],
            model_id=payload["model_id"],
            rollout_index=int(payload["rollout_index"]),
            expected_action=payload["expected_action"],
            checkpoint_sec=float(payload["checkpoint_sec"]),
            window_end_sec=float(payload["window_end_sec"]),
            events=[TraceEvent.from_dict(event) for event in payload["events"]],
            played_audio=[
                PlayedAudio(
                    start_sec=float(audio["start_sec"]),
                    end_sec=float(audio["end_sec"]),
                    rms=float(audio["rms"]),
                )
                for audio in payload["played_audio"]
            ],
            text=payload["text"],
            response_audio=np.zeros(0, dtype=np.float32),
            # Older traces omitted sample_rate; every current writer uses 24 kHz.
            sample_rate=int(payload["sample_rate"])
            if "sample_rate" in payload
            else 24000,
            session=payload.get("session", {}),
            timed_out=bool(payload.get("timed_out", False)),
        )
