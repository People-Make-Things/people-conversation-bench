"""Batch runner for deterministic realtime turn-taking evaluations."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from bench.interact import build_session_config
from bench.metrics.length import (
    first_word_playback_sec,
    model_response_words,
    score_response_length,
)
from bench.metrics.pause import score_pause_recognition
from bench.metrics.recall import score_recall
from bench.metrics.similarity import score_response_similarity
from bench.points import (
    EVAL_TYPES,
    EotScenario,
    LATENCY_EVAL,
    LENGTH_EVAL,
    PAUSE_EVAL,
    RECALL_EVAL_TYPES,
    RecallScenario,
    RolloutScenario,
    SIMILARITY_EVAL,
    eval_type_for_point,
    load_rollout_scenario,
    parse_point_meta,
    read_point_file,
)
from bench.protocol import ContextType
from bench.registry import (
    ModelConfig,
    discover_model_configs,
    load_adapter,
    load_model,
)
from bench.report.summarize import write_json, write_run_identity, write_run_summaries
from bench.results_s3 import ResultsMirror
from bench.rollout import PAUSE_DRAIN_SEC, run_rollout
from bench.trace import SCORED, RolloutTrace
from bench.warmup import raise_open_files, warmup_models
from utils.timed_words import words_text

ALL_MODELS = "all"

BASE_SEED = 424242


def rollout_seed(config: ModelConfig, rollout_index: int) -> int:
    """Seed rollout N reproducibly while keeping rollouts different from each other.

    Override the base with `[session] seed` in model.toml, or -1 to send no seed.
    """
    base = BASE_SEED if config.session.seed is None else config.session.seed
    if base == -1:
        return -1
    return base + rollout_index


def load_manifest_points(path: Path) -> list[Path]:
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    points = manifest.get("points")
    if not isinstance(points, list):
        raise ValueError(f"{path}: expected a points list")
    return [(path.parent / point).resolve() for point in points]


def expand_model_refs(model_refs: Sequence[str]) -> list[str]:
    if ALL_MODELS not in model_refs:
        return list(dict.fromkeys(model_refs))
    return list(discover_model_configs())


def as_manifest_paths(
    manifest: str | Path | Sequence[str | Path] | None,
) -> list[Path] | None:
    if manifest is None:
        return None
    if isinstance(manifest, (str, Path)):
        return [Path(manifest)]
    return [Path(item) for item in manifest]


def collect_point_paths(
    manifests: list[Path] | None,
    point: Path | None,
) -> list[Path]:
    if manifests:
        points: list[Path] = []
        seen: set[Path] = set()
        for manifest in manifests:
            for item in load_manifest_points(manifest):
                if item not in seen:
                    seen.add(item)
                    points.append(item)
        return points
    assert point is not None
    point_path = point / "point.json" if point.is_dir() else point
    return [point_path.resolve()]


def manifest_identity(manifests: list[Path] | None) -> str | list[str] | None:
    if not manifests:
        return None
    values = [str(path) for path in manifests]
    if len(values) == 1:
        return values[0]
    return values


def write_score(rollout_dir: Path, model: str, filename: str, score) -> None:
    payload = score.to_dict()
    payload["model"] = model
    write_json(rollout_dir / filename, payload)


SCORE_FILES = {
    PAUSE_EVAL: "score.json",
    LENGTH_EVAL: "score.json",
    SIMILARITY_EVAL: "similarity.json",
    **{name: "score.json" for name in RECALL_EVAL_TYPES},
}


def score_metric(metric: str, trace: RolloutTrace, scenario: RolloutScenario, words):
    if metric == PAUSE_EVAL:
        return score_pause_recognition(trace)
    if metric == LENGTH_EVAL:
        assert isinstance(scenario, EotScenario)
        return score_response_length(
            trace,
            human_duration_sec=scenario.reference.duration_sec,
            human_word_count=scenario.reference.word_count,
            model_word_count=len(words),
            first_word_sec=first_word_playback_sec(trace, words),
            human_text=scenario.reference.text,
        )
    if metric == SIMILARITY_EVAL:
        assert isinstance(scenario, EotScenario)
        return score_response_similarity(
            trace,
            human_text=scenario.reference.text,
            model_text=words_text(words),
        )
    if metric in RECALL_EVAL_TYPES:
        assert isinstance(scenario, RecallScenario)
        return score_recall(
            trace,
            question=scenario.question,
            gold_answer=scenario.gold_answer,
            history_target_sec=scenario.history_target_sec,
            history_sec=scenario.history_sec,
            transcript=words_text(words),
            conversation=scenario.conversation,
            first_word_sec=first_word_playback_sec(trace, words),
        )
    raise ValueError(f"unknown metric: {metric}")


def write_trace_scores(
    rollout_dir: Path,
    model: str,
    trace: RolloutTrace,
    scenario: RolloutScenario,
) -> None:
    # A rollout with no model audio is unscored, so there is nothing to
    # transcribe and no reason to pay for a Whisper pass on silence.
    needs_words = any(metric != PAUSE_EVAL for metric in scenario.metrics)
    words = model_response_words(trace) if needs_words and trace.status == SCORED else []
    for metric in scenario.metrics:
        write_score(
            rollout_dir,
            model,
            SCORE_FILES[metric],
            score_metric(metric, trace, scenario, words),
        )


def record_rollout_error(
    rollout_dir: Path,
    point_type: str,
    model_ref: str,
    error: Exception,
    *,
    label: str,
) -> None:
    print(f"{label} {rollout_dir} failed: {error!r}", flush=True)
    write_json(
        rollout_dir / "error.json",
        {
            "eval": point_type,
            "model": model_dir_name(model_ref),
            "error": repr(error),
        },
    )


async def run_one_rollout(
    model_ref: str,
    config: ModelConfig,
    output_dir: Path,
    point_path: Path,
    point_meta: dict,
    point_id: str,
    point_type: str,
    rollout_index: int,
    pause_drain_sec: float,
    semaphore: asyncio.Semaphore,
    kinds: tuple[ContextType, ...],
    mirror: ResultsMirror | None,
) -> None:
    rollout_dir = rollout_output_dir(
        output_dir,
        point_type,
        model_ref,
        point_id,
        rollout_index,
    )
    # A restart of the same --output must not redo scored work. error.json is
    # retried: those holes are why a restart happens.
    if (rollout_dir / "score.json").is_file():
        return
    trace = None
    async with semaphore:
        try:
            scenario = load_rollout_scenario(
                point_path, kinds=kinds, meta=point_meta
            )
            if not scenario.text_prompt:
                raise ValueError(
                    f"{point_path}: no text_prompt, so the model would run with no "
                    "identity frame while assistant history is forced as its own speech"
                )
            error_path = rollout_dir / "error.json"
            if error_path.is_file():
                error_path.unlink()
            model = load_adapter(config)
            session_config = build_session_config(
                config,
                voice_prompt=None,
                text_prompt=scenario.text_prompt,
                seed=rollout_seed(config, rollout_index),
            )
            trace = await run_rollout(
                model,
                config,
                session_config,
                scenario,
                rollout_dir,
                rollout_index=rollout_index,
                pause_drain_sec=pause_drain_sec,
            )
        except Exception as error:
            record_rollout_error(
                rollout_dir, point_type, model_ref, error, label="rollout"
            )
    if trace is not None:
        # Scoring reaches the network (recall judge) and can raise like the
        # rollout itself; contain it the same way so one bad score does not
        # abort the model's whole batch. Partial score files are removed so
        # error.json keeps meaning "retry me" to a restart.
        try:
            write_trace_scores(
                rollout_dir, model_dir_name(model_ref), trace, scenario
            )
        except Exception as error:
            for name in ("score.json", "similarity.json"):
                (rollout_dir / name).unlink(missing_ok=True)
            record_rollout_error(
                rollout_dir, point_type, model_ref, error, label="scoring"
            )
    if mirror is not None:
        await mirror.upload_dir(rollout_dir)


async def run_model_jobs(
    model_ref: str,
    output_dir: Path,
    point_paths: list[Path],
    rollouts: int,
    eval_type: str | None,
    pause_drain_sec: float,
    mirror: ResultsMirror | None,
) -> None:
    config, probe = load_model(model_ref)
    jobs: list[tuple[Path, dict, str, str, int]] = []
    for point_path in dict.fromkeys(point_paths):
        _point_json, point_dir, point_meta = read_point_file(point_path)
        point_id = parse_point_meta(point_meta, default_id=point_dir.name).point_id
        point_type = eval_type_for_point(point_meta)
        if point_type is None:
            raise ValueError(
                f"{point_path}: no expected action, so no eval applies"
            )
        if eval_type is not None and point_type != eval_type:
            continue
        for rollout_index in range(rollouts):
            jobs.append((point_path, point_meta, point_id, point_type, rollout_index))

    semaphore = asyncio.Semaphore(config.max_concurrency)
    # One failed rollout must not discard the rollouts already scored,
    # so it is recorded on disk and the batch continues. Audio is loaded
    # inside each job so a 500-point batch does not keep every history in RAM.
    await asyncio.gather(
        *[
            run_one_rollout(
                model_ref,
                config,
                output_dir,
                point_path,
                point_meta,
                point_id,
                point_type,
                rollout_index,
                pause_drain_sec,
                semaphore,
                probe.supported_contexts,
                mirror,
            )
            for point_path, point_meta, point_id, point_type, rollout_index in jobs
        ]
    )


async def run_suite(
    model_refs: list[str],
    output_dir: Path,
    manifest: str | Path | Sequence[str | Path] | None = None,
    point: Path | None = None,
    rollouts: int = 3,
    eval_type: str | None = None,
    pause_drain_sec: float = PAUSE_DRAIN_SEC,
    warmup: bool = True,
    mirror: ResultsMirror | None = None,
) -> tuple[dict, list[Path]]:
    if eval_type is not None and eval_type not in EVAL_TYPES:
        raise ValueError(f"unknown eval type: {eval_type}")
    manifests = as_manifest_paths(manifest)
    if (not manifests) == (point is None):
        raise ValueError("pass a manifest (one or more) or a point, not both")
    point_paths = collect_point_paths(manifests, point)

    refs = expand_model_refs(model_refs)
    raise_open_files()
    if warmup:
        await warmup_models(refs)
    print(
        f"eval rollouts starting models={refs} points={len(point_paths)}",
        flush=True,
    )
    write_run_identity(
        output_dir,
        manifest=manifest_identity(manifests),
        point=str(point) if point is not None else None,
    )
    await asyncio.gather(
        *[
            run_model_jobs(
                model_ref,
                output_dir,
                point_paths,
                rollouts,
                eval_type,
                pause_drain_sec,
                mirror,
            )
            for model_ref in refs
        ]
    )
    summaries = write_run_summaries(
        output_dir,
        manifest=manifest_identity(manifests),
        point=str(point) if point is not None else None,
    )
    if mirror is not None:
        await mirror.upload_summaries()
    return summaries


def model_dir_name(model_ref: str) -> str:
    return model_ref.removeprefix("@").replace("/", "_")


def default_output_dir() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("data/results") / timestamp


def rollout_output_dir(
    run_dir: Path,
    eval_type: str,
    model_ref: str,
    point_id: str,
    rollout_index: int,
) -> Path:
    return (
        run_dir
        / eval_type
        / model_dir_name(model_ref)
        / point_id
        / f"rollout_{rollout_index + 1:03d}"
    )


def run(
    model_refs: list[str],
    manifest: str | Path | Sequence[str | Path] | None = None,
    point: str | Path | None = None,
    output_dir: str | Path | None = None,
    rollouts: int = 3,
    eval_type: str | None = None,
    pause_drain_sec: float = PAUSE_DRAIN_SEC,
    warmup: bool = True,
) -> None:
    destination = (
        Path(output_dir) if output_dir is not None else default_output_dir()
    )
    results, plots = asyncio.run(
        run_suite(
            model_refs,
            destination,
            manifest=manifest,
            point=Path(point) if point is not None else None,
            rollouts=rollouts,
            eval_type=eval_type,
            pause_drain_sec=pause_drain_sec,
            warmup=warmup,
            mirror=ResultsMirror(destination),
        )
    )
    total = sum(results[name]["rollout_count"] for name in EVAL_TYPES)
    skipped = sum(
        len(results[name][key])
        for name in EVAL_TYPES
        for key in ("unscored", "errors")
    )
    parts: list[str] = []
    for name in EVAL_TYPES:
        metrics = results[name]
        if not metrics["rollout_count"]:
            continue
        if name == LENGTH_EVAL:
            parts.append(
                "response-length "
                f"duration_ratio={metrics['mean_duration_ratio']} "
                f"word_ratio={metrics['mean_word_ratio']}"
            )
            continue
        parts.append(f"{name} mean={metrics['mean_score']}")
    similarity = results[SIMILARITY_EVAL]
    if similarity["rollout_count"]:
        parts.append(
            "response-similarity "
            f"token_f1={similarity['mean_token_f1']} "
            f"cosine={similarity['mean_embedding_cosine']}"
        )
    latency = results[LATENCY_EVAL]
    if latency["rollout_count"]:
        parts.append(
            "response-latency "
            + ", ".join(
                f"{name}={metrics['mean_latency_ms']}"
                for name, metrics in latency["by_eval"].items()
            )
        )
    parts.append(
        f"unscored={sum(len(results[name]['unscored']) for name in EVAL_TYPES)}, "
        f"errors={sum(len(results[name]['errors']) for name in EVAL_TYPES)}, "
        f"timed_out={len(results[LENGTH_EVAL]['timed_out'])}"
    )
    suffix = f" ({'; '.join(parts)})" if parts else ""
    extra = f"; {', '.join(path.name for path in plots)}" if plots else ""
    print(f"Wrote {total} scored rollout(s) to {destination}{suffix}{extra}")
    if skipped:
        print(f"{skipped} rollout(s) were not scored; see results.json", flush=True)
