"""Load scored and failed rollouts from a run directory once."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from bench.metrics.length import ResponseLengthScore
from bench.metrics.pause import PauseRecognitionScore
from bench.metrics.recall import RecallScore
from bench.metrics.similarity import ResponseSimilarityScore
from bench.points import EVAL_TYPES


Score = (
    PauseRecognitionScore
    | ResponseLengthScore
    | ResponseSimilarityScore
    | RecallScore
)
SCORE_TYPES = {
    PauseRecognitionScore.eval: PauseRecognitionScore,
    ResponseLengthScore.eval: ResponseLengthScore,
    ResponseSimilarityScore.eval: ResponseSimilarityScore,
    "conversation-recall": RecallScore,
    "fact-recall": RecallScore,
}


@dataclass(frozen=True)
class RunArtifact:
    path: Path
    model: str
    eval_type: str
    score: Score | None = None
    error: str | None = None


def parse_score(kind: str, data: dict) -> Score:
    score_type = SCORE_TYPES.get(kind)
    if score_type is None:
        raise ValueError(f"unknown eval type: {kind}")
    return score_type.from_dict(data)


def parse_artifact(path: Path) -> RunArtifact | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    kind = data.get("eval")
    model = str(data.get("model") or "")
    if not model:
        return None
    if path.name == "similarity.json":
        if kind != "response-similarity":
            return None
        return RunArtifact(
            path, model, "response-similarity", score=parse_score(str(kind), data)
        )
    if kind not in EVAL_TYPES:
        return None
    if path.name == "error.json":
        return RunArtifact(
            path, model, str(kind), error=str(data.get("error", ""))
        )
    return RunArtifact(path, model, str(kind), score=parse_score(str(kind), data))


def load_run_artifacts(run_dir: Path) -> list[RunArtifact]:
    artifacts: list[RunArtifact] = []
    for pattern in (
        "**/rollout_*/error.json",
        "**/rollout_*/score.json",
        "**/rollout_*/similarity.json",
    ):
        for path in sorted(run_dir.glob(pattern)):
            artifact = parse_artifact(path)
            if artifact is not None:
                artifacts.append(artifact)
    return artifacts
