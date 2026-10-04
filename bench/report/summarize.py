"""Rebuild run summaries and figures from stored rollout artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bench.metrics.length import ResponseLengthScore
from bench.metrics.pause import PauseRecognitionScore
from bench.metrics.recall import RecallScore
from bench.metrics.similarity import ResponseSimilarityScore
from bench.points import (
    EVAL_TYPES,
    LATENCY_EVAL,
    LENGTH_EVAL,
    RECALL_EVAL_TYPES,
    REPORT_SECTIONS,
)
from bench.report.artifacts import RunArtifact, load_run_artifacts
from bench.report.plots import (
    LENGTH_AVERAGE_NAME,
    LENGTH_NAME,
    PAUSE_NAME,
    RECALL_NAME,
    SIMILARITY_NAME,
    write_run_dir_plots,
)
from bench.trace import SCORED

INDEX_NAME = "index.html"
ROOT_FIGURES = (
    PAUSE_NAME,
    LENGTH_NAME,
    LENGTH_AVERAGE_NAME,
    RECALL_NAME,
    SIMILARITY_NAME,
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def pause_recognition_results(
    scores: list[PauseRecognitionScore],
    unscored: list[dict],
    errors: list[dict],
) -> dict:
    return {
        "rollout_count": len(scores),
        "mean_score": mean_or_none([float(score.score) for score in scores]),
        # A pause score says only that the model did not utter words through
        # the hold, which a model voicing without words can pass. Reported
        # here so the mean cannot be read without it.
        "wordless_phonation": sum(1 for score in scores if score.wordless_phonation),
        "unscored": unscored,
        "errors": errors,
    }


def response_similarity_results(
    scores: list[ResponseSimilarityScore],
    unscored: list[dict],
    errors: list[dict],
    timed_out: list[dict],
) -> dict:
    return {
        "rollout_count": len(scores),
        "mean_token_f1": mean_or_none([score.token_f1 for score in scores]),
        "mean_embedding_cosine": mean_or_none(
            [score.embedding_cosine for score in scores]
        ),
        "unscored": unscored,
        "errors": errors,
        "timed_out": timed_out,
    }


def response_length_results(
    scores: list[ResponseLengthScore],
    unscored: list[dict],
    errors: list[dict],
    timed_out: list[dict],
) -> dict:
    duration_ratios = [
        score.duration_ratio for score in scores if score.duration_ratio is not None
    ]
    word_ratios = [
        score.word_ratio for score in scores if score.word_ratio is not None
    ]
    return {
        "rollout_count": len(scores),
        "mean_duration_ratio": mean_or_none(duration_ratios),
        "mean_word_ratio": mean_or_none(word_ratios),
        "mean_duration_delta_sec": mean_or_none(
            [score.duration_delta_sec for score in scores]
        ),
        "mean_word_delta": mean_or_none([float(score.word_delta) for score in scores]),
        "unscored": unscored,
        "errors": errors,
        "timed_out": timed_out,
    }


def latency_eval_results(values: list[float]) -> dict:
    return {
        "rollout_count": len(values),
        "mean_latency_ms": mean_or_none(values),
        "median_latency_ms": float(np.median(values)) if values else None,
        "p90_latency_ms": float(np.percentile(values, 90)) if values else None,
    }


def response_latency_results(
    by_eval: dict[str, list[float]],
    wordless_phonation: int,
    unscored: list[dict],
    errors: list[dict],
) -> dict:
    return {
        "rollout_count": sum(len(values) for values in by_eval.values()),
        "by_eval": {
            name: latency_eval_results(values)
            for name, values in sorted(by_eval.items())
        },
        "wordless_phonation": wordless_phonation,
        "unscored": unscored,
        "errors": errors,
    }


def response_latency_from_artifacts(artifacts: list[RunArtifact]) -> dict:
    by_eval: dict[str, list[float]] = {}
    unscored: list[dict] = []
    wordless = 0
    for artifact in artifacts:
        score = artifact.score
        if (
            score is None
            or score.status != SCORED
            or not isinstance(score, (ResponseLengthScore, RecallScore))
        ):
            continue
        eval_type = (
            LENGTH_EVAL
            if isinstance(score, ResponseLengthScore)
            else artifact.eval_type
        )
        rollout = str(artifact.path.parent)
        if score.latency_ms is not None:
            by_eval.setdefault(eval_type, []).append(float(score.latency_ms))
            continue
        if score.wordless_phonation:
            wordless += 1
        unscored.append(
            {
                "rollout": rollout,
                "status": (
                    "wordless_phonation"
                    if score.wordless_phonation
                    else "empty_response"
                ),
            }
        )
    return response_latency_results(by_eval, wordless, unscored, [])


def recall_results(
    scores: list[RecallScore],
    unscored: list[dict],
    errors: list[dict],
) -> dict:
    by_history: dict[str, list[float]] = {}
    for score in scores:
        key = f"{score.history_target_sec:g}"
        by_history.setdefault(key, []).append(float(score.score))
    return {
        "rollout_count": len(scores),
        "mean_score": mean_or_none([float(score.score) for score in scores]),
        "by_history_sec": {
            key: mean_or_none(values)
            for key, values in sorted(by_history.items(), key=lambda item: float(item[0]))
        },
        "unscored": unscored,
        "errors": errors,
    }


def eval_has_output(metrics: dict) -> bool:
    return any(
        metrics.get(key) for key in ("rollout_count", "unscored", "errors", "timed_out")
    )


def summarize_artifacts(
    artifacts: list[RunArtifact],
    *,
    manifest: str | list[str] | None,
    point: str | None,
) -> dict:
    pause_scores: list[PauseRecognitionScore] = []
    length_scores: list[ResponseLengthScore] = []
    similarity_scores: list[ResponseSimilarityScore] = []
    recall_scores: dict[str, list[RecallScore]] = {
        name: [] for name in RECALL_EVAL_TYPES
    }
    unscored: dict[str, list[dict]] = {name: [] for name in REPORT_SECTIONS}
    errors: dict[str, list[dict]] = {name: [] for name in REPORT_SECTIONS}
    timed_out_length: list[dict] = []
    timed_out_similarity: list[dict] = []
    models: list[str] = []

    for artifact in artifacts:
        if artifact.model not in models:
            models.append(artifact.model)
        rollout = str(artifact.path.parent)
        if artifact.error is not None:
            errors[artifact.eval_type].append(
                {"rollout": rollout, "error": artifact.error}
            )
            continue
        assert artifact.score is not None
        if artifact.score.status != SCORED:
            unscored[artifact.eval_type].append(
                {"rollout": rollout, "status": artifact.score.status}
            )
            continue
        if isinstance(artifact.score, PauseRecognitionScore):
            pause_scores.append(artifact.score)
        elif isinstance(artifact.score, RecallScore):
            recall_scores[artifact.eval_type].append(artifact.score)
        elif isinstance(artifact.score, ResponseLengthScore):
            # A rollout the drain cap cut off still scores: a model that will
            # not stop is a finding, not missing data. Its duration is censored
            # at the cap (first words and transcript are unaffected), and
            # timed_out records which values in the means are lower bounds.
            length_scores.append(artifact.score)
            if artifact.score.timed_out:
                timed_out_length.append(
                    {
                        "rollout": rollout,
                        "model_duration_sec": artifact.score.model_duration_sec,
                    }
                )
        elif isinstance(artifact.score, ResponseSimilarityScore):
            similarity_scores.append(artifact.score)
            if artifact.score.timed_out:
                timed_out_similarity.append(
                    {
                        "rollout": rollout,
                        "token_f1": artifact.score.token_f1,
                        "embedding_cosine": artifact.score.embedding_cosine,
                    }
                )
        else:
            raise TypeError(f"unknown score type: {type(artifact.score)}")

    return {
        "models": models,
        "manifest": manifest,
        "point": point,
        "pause-recognition": pause_recognition_results(
            pause_scores,
            unscored["pause-recognition"],
            errors["pause-recognition"],
        ),
        "response-length": response_length_results(
            length_scores,
            unscored["response-length"],
            errors["response-length"],
            timed_out_length,
        ),
        "response-similarity": response_similarity_results(
            similarity_scores,
            unscored["response-similarity"],
            errors["response-similarity"],
            timed_out_similarity,
        ),
        LATENCY_EVAL: response_latency_from_artifacts(artifacts),
        **{
            name: recall_results(recall_scores[name], unscored[name], errors[name])
            for name in RECALL_EVAL_TYPES
        },
    }


def load_run_identity(
    run_dir: Path,
) -> tuple[str | list[str] | None, str | None]:
    for name in ("run.json", "results.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("manifest"), data.get("point")
    return None, None


def write_run_identity(
    run_dir: Path, *, manifest: str | list[str] | None, point: str | None
) -> None:
    write_json(run_dir / "run.json", {"manifest": manifest, "point": point})


def write_run_index(run_dir: Path) -> Path | None:
    figures = [name for name in ROOT_FIGURES if (run_dir / name).is_file()]
    if not figures:
        return None
    items = "\n".join(
        f'<li><a href="{name}"><img src="{name}" alt="{name}"></a></li>'
        for name in figures
    )
    path = run_dir / INDEX_NAME
    path.write_text(
        "<!doctype html>\n"
        f"<title>{run_dir.name}</title>\n"
        "<style>body{font-family:serif;margin:1.5rem}img{max-width:40rem;height:auto}"
        "li{list-style:none;margin:1.2rem 0}</style>\n"
        f"<h1>{run_dir.name}</h1>\n"
        f"<ul>\n{items}\n</ul>\n",
        encoding="utf-8",
    )
    return path


def write_run_summaries(
    run_dir: Path,
    *,
    manifest: str | list[str] | None,
    point: str | None,
) -> tuple[dict, list[Path]]:
    if manifest is None and point is None:
        manifest, point = load_run_identity(run_dir)
    write_run_identity(run_dir, manifest=manifest, point=point)
    artifacts = load_run_artifacts(run_dir)
    results = summarize_artifacts(artifacts, manifest=manifest, point=point)
    identity = {
        "models": results["models"],
        "manifest": results["manifest"],
        "point": results["point"],
    }
    write_json(run_dir / "results.json", results)
    for name in EVAL_TYPES:
        metrics = results[name]
        if eval_has_output(metrics):
            write_json(
                run_dir / name / "results.json",
                {"eval": name, **identity, **metrics},
            )
    paths = write_run_dir_plots(run_dir, artifacts)
    index = write_run_index(run_dir)
    if index is not None:
        paths.append(index)
    return results, paths
