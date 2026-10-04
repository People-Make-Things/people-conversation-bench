"""Publication-style summary figures for an eval run directory."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench.metrics.length import ResponseLengthScore
from bench.metrics.pause import (
    HOLD,
    NO_AUDIO,
    TAKEOVER,
    WORDLESS,
    PauseRecognitionScore,
    pause_outcome,
)
from bench.metrics.recall import RecallScore
from bench.metrics.similarity import ResponseSimilarityScore
from bench.points import LENGTH_EVAL, RECALL_EVAL_TYPES
from bench.report.artifacts import RunArtifact, load_run_artifacts
from bench.trace import NO_MODEL_AUDIO, SCORED

PAUSE_NAME = "pause.png"
LENGTH_NAME = "length.png"
LENGTH_AVERAGE_NAME = "length-average.png"
RECALL_NAME = "recall.png"
SIMILARITY_NAME = "similarity.png"
LATENCY_NAME = "latency.png"
MODEL_COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2", "#72b7b2")
MODEL_MARKERS = ("o", "s", "D", "^", "v", "P")
PAUSE_BAR_COLORS = {
    TAKEOVER: "#e45756",
    HOLD: "#54a24b",
    WORDLESS: "#f58518",
    NO_AUDIO: "#c7c7c7",
}
STYLE = {
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
}


class PausePoint(NamedTuple):
    model: str
    score: PauseRecognitionScore


class LengthPoint(NamedTuple):
    model: str
    score: ResponseLengthScore


class RecallPoint(NamedTuple):
    model: str
    eval_type: str
    score: RecallScore


class SimilarityPoint(NamedTuple):
    model: str
    score: ResponseSimilarityScore


class LatencyPoint(NamedTuple):
    model: str
    eval_type: str
    latency_ms: float


def unique_names(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def model_colors(names: list[str]) -> dict[str, str]:
    return {
        name: MODEL_COLORS[index % len(MODEL_COLORS)]
        for index, name in enumerate(unique_names(names))
    }


def plot_rows(
    artifacts: list[RunArtifact],
) -> tuple[
    list[PausePoint],
    list[LengthPoint],
    list[RecallPoint],
    list[SimilarityPoint],
]:
    pause: list[PausePoint] = []
    length: list[LengthPoint] = []
    recall: list[RecallPoint] = []
    similarity: list[SimilarityPoint] = []
    for artifact in artifacts:
        score = artifact.score
        if score is None:
            continue
        if (
            isinstance(score, PauseRecognitionScore)
            and score.status in (SCORED, NO_MODEL_AUDIO)
        ):
            pause.append(PausePoint(artifact.model, score))
        elif isinstance(score, ResponseLengthScore) and score.status == SCORED:
            length.append(LengthPoint(artifact.model, score))
        elif (
            isinstance(score, RecallScore)
            and score.status == SCORED
            and score.score is not None
        ):
            recall.append(RecallPoint(artifact.model, artifact.eval_type, score))
        elif isinstance(score, ResponseSimilarityScore) and score.status == SCORED:
            similarity.append(SimilarityPoint(artifact.model, score))
    return pause, length, recall, similarity


def latency_points(
    length: list[LengthPoint],
    recall: list[RecallPoint],
) -> list[LatencyPoint]:
    return [
        *(
            LatencyPoint(row.model, LENGTH_EVAL, float(row.score.latency_ms))
            for row in length
            if row.score.latency_ms is not None
        ),
        *(
            LatencyPoint(row.model, row.eval_type, float(row.score.latency_ms))
            for row in recall
            if row.score.latency_ms is not None
        ),
    ]


def jittered_x(xs: list[float], lim: float) -> list[float]:
    groups: dict[float, list[int]] = {}
    for index, x in enumerate(xs):
        groups.setdefault(x, []).append(index)
    out = list(xs)
    span = max(lim * 0.02, 8.0)
    for x, indexes in groups.items():
        if len(indexes) == 1:
            continue
        offsets = np.linspace(-span / 2, span / 2, len(indexes))
        for index, offset in zip(indexes, offsets):
            out[index] = x + offset
    return out


def identity_limits(xs: list[float], ys: list[float]) -> float:
    peak = max([0.0, *xs, *ys])
    return peak * 1.08 if peak > 0 else 1.0


def grouped(
    rows: list[PausePoint] | list[LengthPoint] | list[RecallPoint] | list[SimilarityPoint],
) -> dict[str, list]:
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(row.model, []).append(row)
    return groups


def identity_axes(
    ax,
    xs: list[float],
    ys: list[float],
    xlabel: str,
    ylabel: str,
    lim: float | None = None,
) -> float:
    if lim is None:
        lim = identity_limits(xs, ys)
    ax.plot([0.0, lim], [0.0, lim], color="0.55", lw=0.8, zorder=0)
    ax.set_xlim(0.0, lim)
    ax.set_ylim(0.0, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return lim


def scatter_vs_human(
    ax,
    human: list[float],
    predicted: list[float],
    xlabel: str,
    ylabel: str,
    model_ids: list[str],
    colors: dict[str, str],
    lim: float | None = None,
    marked: bool = False,
) -> None:
    identity_axes(ax, human, predicted, xlabel, ylabel, lim=lim)
    names = unique_names(model_ids)
    for index, name in enumerate(names):
        xs = [x for x, model in zip(human, model_ids) if model == name]
        ys = [y for y, model in zip(predicted, model_ids) if model == name]
        ax.scatter(
            xs,
            ys,
            s=42 if marked else 22,
            c=colors[name],
            marker=MODEL_MARKERS[index % len(MODEL_MARKERS)] if marked else "o",
            edgecolors="none",
            zorder=2,
            label=name if marked else None,
        )


def pause_limit(rows: list[PausePoint]) -> float:
    waits = [
        float(row.score.model_wait_from_pause_start_ms)
        for row in rows
        if row.score.model_wait_from_pause_start_ms is not None
        and row.score.status != NO_MODEL_AUDIO
    ]
    return identity_limits(
        [float(row.score.labeled_pause_duration_ms or 0.0) for row in rows],
        waits,
    )


def pause_title(rows: list[PausePoint], name: str) -> str:
    scored = [
        row
        for row in rows
        if row.score.status == SCORED and row.score.score is not None
    ]
    dead = sum(1 for row in rows if row.score.status == NO_MODEL_AUDIO)
    hold_rate = float(np.mean([row.score.score for row in scored])) if scored else None
    rate = "—" if hold_rate is None else f"{hold_rate:.2f}"
    title = f"{name}   hold = {rate}  (n={len(scored)}"
    if dead:
        title += f", {dead} no audio"
    return title + ")"


def draw_pause(
    ax,
    rows: list[PausePoint],
    colors: dict[str, str],
    lim: float,
) -> None:
    raw_xs = [float(row.score.labeled_pause_duration_ms or 0.0) for row in rows]
    identity_axes(
        ax,
        raw_xs,
        [
            float(row.score.model_wait_from_pause_start_ms)
            for row in rows
            if row.score.model_wait_from_pause_start_ms is not None
            and row.score.status != NO_MODEL_AUDIO
        ],
        "Labeled pause (ms)",
        "Model wait from pause start (ms)",
        lim=lim,
    )
    points = [
        (
            row.model,
            pause_outcome(row.score),
            x,
            (
                None
                if row.score.model_wait_from_pause_start_ms is None
                or row.score.status == NO_MODEL_AUDIO
                else float(row.score.model_wait_from_pause_start_ms)
            ),
        )
        for row, x in zip(rows, jittered_x(raw_xs, lim))
    ]
    series = (
        (TAKEOVER, "x", {"s": 22, "linewidths": 0.8}),
        (HOLD, "o", {"s": 22, "linewidths": 0.0}),
        (WORDLESS, "^", {"s": 22, "linewidths": 0.0, "clip_on": False}),
        (NO_AUDIO, "D", {"s": 20, "linewidths": 0.7, "facecolors": "none", "clip_on": False}),
    )
    for label, marker, kwargs in series:
        for model in unique_names([row.model for row in rows]):
            xs = [
                x
                for point_model, kind, x, _y in points
                if kind == label and point_model == model
            ]
            if not xs:
                continue
            ys = [
                lim if y is None else y
                for point_model, kind, _x, y in points
                if kind == label and point_model == model
            ]
            scatter_kwargs = dict(kwargs)
            if label == NO_AUDIO:
                scatter_kwargs["edgecolors"] = colors[model]
            else:
                scatter_kwargs["c"] = colors[model]
            ax.scatter(xs, ys, marker=marker, zorder=2, **scatter_kwargs)


def save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def recall_mean(rows: list[RecallPoint]) -> float | None:
    scores = [float(row.score.score) for row in rows if row.score.score is not None]
    if not scores:
        return None
    return float(np.mean(scores))


def draw_recall(ax, rows: list[RecallPoint], colors: dict[str, str]) -> None:
    ax.set_xlabel("History (s)")
    ax.set_ylabel("Recall")
    ax.set_ylim(-0.08, 1.08)
    ax.set_xticks([30, 60, 90, 120])
    ax.set_xlim(16, 134)
    names = unique_names([row.model for row in rows])
    offsets = (
        np.linspace(-10.0, 10.0, len(names)) if len(names) > 1 else np.array([0.0])
    )
    for index, name in enumerate(names):
        group = [row for row in rows if row.model == name]
        xs = sorted({row.score.history_target_sec for row in group})
        means = []
        for x in xs:
            values = [
                float(row.score.score)
                for row in group
                if row.score.history_target_sec == x and row.score.score is not None
            ]
            means.append(float(np.mean(values)) if values else float("nan"))
        ax.plot(
            [x + offsets[index] for x in xs],
            means,
            color=colors[name],
            marker=MODEL_MARKERS[index % len(MODEL_MARKERS)],
            lw=1.2,
            ms=4,
            label=name,
        )
    if len(names) > 1:
        ax.legend(
            loc="lower left",
            frameon=True,
            fancybox=False,
            edgecolor="0.75",
            framealpha=1,
            fontsize=7,
        )


def write_recall_figure(
    rows: list[RecallPoint],
    path: Path,
    colors: dict[str, str],
) -> None:
    eval_types = unique_names([row.eval_type for row in rows]) or list(RECALL_EVAL_TYPES)
    fig, axes = plt.subplots(
        1,
        len(eval_types),
        figsize=(3.2 * len(eval_types), 3.1),
        squeeze=False,
    )
    for ax, eval_type in zip(axes[0], eval_types):
        typed = [row for row in rows if row.eval_type == eval_type]
        draw_recall(ax, typed, colors)
        mean = recall_mean(typed)
        rate = "—" if mean is None else f"{mean:.2f}"
        ax.set_title(f"{eval_type}   mean={rate}  (n={len(typed)})")
    save_figure(fig, path)


def write_pause_figure(
    rows: list[PausePoint],
    path: Path,
    colors: dict[str, str],
    lim: float,
) -> None:
    names = unique_names([row.model for row in rows])
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    draw_pause(ax, rows, colors, lim)
    ax.set_title(pause_title(rows, names[0] if names else "Pause"))
    save_figure(fig, path)


def write_pause_bars(rows: list[PausePoint], path: Path) -> None:
    names = unique_names([row.model for row in rows])
    counts = {name: {label: 0 for label in PAUSE_BAR_COLORS} for name in names}
    for row in rows:
        counts[row.model][pause_outcome(row.score)] += 1
    totals = [sum(counts[name].values()) or 1 for name in names]
    fig, ax = plt.subplots(figsize=(max(3.4, 1.2 * len(names) + 1.4), 3.4))
    bottom = np.zeros(len(names))
    for label, color in PAUSE_BAR_COLORS.items():
        heights = [
            counts[name][label] / total for name, total in zip(names, totals)
        ]
        ax.bar(names, heights, bottom=bottom, color=color, width=0.72, label=label)
        bottom = bottom + heights
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share of pause rollouts")
    ax.set_title("Pause outcomes")
    ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="0.75",
        framealpha=1,
        fontsize=7,
    )
    ax.tick_params(axis="x", labelrotation=20)
    save_figure(fig, path)


def length_caption(rows: list[LengthPoint], name: str) -> str:
    duration_values = [
        row.score.duration_ratio for row in rows if row.score.duration_ratio is not None
    ]
    word_values = [
        row.score.word_ratio for row in rows if row.score.word_ratio is not None
    ]
    dur_label = f"{float(np.mean(duration_values)):.2f}" if duration_values else "—"
    word_label = f"{float(np.mean(word_values)):.2f}" if word_values else "—"
    return f"{name}   duration ×{dur_label}   words ×{word_label}   (n={len(rows)})"


def length_limits(rows: list[LengthPoint]) -> tuple[float, float]:
    return (
        identity_limits(
            [row.score.human_duration_sec for row in rows],
            [row.score.model_duration_sec for row in rows],
        ),
        identity_limits(
            [float(row.score.human_word_count) for row in rows],
            [float(row.score.model_word_count) for row in rows],
        ),
    )


def draw_length(
    ax_duration,
    ax_words,
    rows: list[LengthPoint],
    colors: dict[str, str],
    duration_lim: float,
    word_lim: float,
) -> None:
    model_ids = [row.model for row in rows]
    scatter_vs_human(
        ax_duration,
        [row.score.human_duration_sec for row in rows],
        [row.score.model_duration_sec for row in rows],
        "Human duration (s)",
        "Model duration (s)",
        model_ids,
        colors,
        lim=duration_lim,
    )
    scatter_vs_human(
        ax_words,
        [float(row.score.human_word_count) for row in rows],
        [float(row.score.model_word_count) for row in rows],
        "Human words",
        "Model words",
        model_ids,
        colors,
        lim=word_lim,
    )


def write_length_figure(
    rows: list[LengthPoint],
    path: Path,
    colors: dict[str, str],
    duration_lim: float,
    word_lim: float,
) -> None:
    names = unique_names([row.model for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))
    draw_length(axes[0], axes[1], rows, colors, duration_lim, word_lim)
    fig.suptitle(length_caption(rows, names[0] if names else "Response length"), y=1.02)
    save_figure(fig, path)


def write_length_grid(
    rows: list[LengthPoint],
    path: Path,
    colors: dict[str, str],
    duration_lim: float,
    word_lim: float,
) -> None:
    names = unique_names([row.model for row in rows])
    if len(names) <= 1:
        write_length_figure(rows, path, colors, duration_lim, word_lim)
        return
    groups = grouped(rows)
    fig, axes = plt.subplots(len(names), 2, figsize=(6.4, 3.0 * len(names)), squeeze=False)
    for index, name in enumerate(names):
        draw_length(
            axes[index][0],
            axes[index][1],
            groups[name],
            colors,
            duration_lim,
            word_lim,
        )
        axes[index][0].set_title(length_caption(groups[name], name), loc="left")
    fig.tight_layout()
    save_figure(fig, path)


def model_means(rows: list[LengthPoint], attr: str) -> list[float]:
    groups = grouped(rows)
    return [
        float(np.mean([getattr(row.score, attr) for row in groups[name]]))
        for name in unique_names([row.model for row in rows])
    ]


def write_length_average(
    rows: list[LengthPoint],
    path: Path,
    colors: dict[str, str],
) -> None:
    names = unique_names([row.model for row in rows])
    human_dur = model_means(rows, "human_duration_sec")
    model_dur = model_means(rows, "model_duration_sec")
    human_words = model_means(rows, "human_word_count")
    model_words = model_means(rows, "model_word_count")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.1))
    scatter_vs_human(
        axes[0],
        human_dur,
        model_dur,
        "Human duration (s)",
        "Model duration (s)",
        names,
        colors,
        marked=True,
    )
    scatter_vs_human(
        axes[1],
        human_words,
        model_words,
        "Human words",
        "Model words",
        names,
        colors,
        marked=True,
    )
    if len(names) > 1:
        axes[1].legend(
            *axes[0].get_legend_handles_labels(),
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="0.75",
            framealpha=1,
            fontsize=7,
        )
    fig.suptitle("Response length (mean per model)", y=1.02)
    save_figure(fig, path)


def similarity_caption(rows: list[SimilarityPoint], name: str) -> str:
    f1 = [row.score.token_f1 for row in rows]
    cosine = [row.score.embedding_cosine for row in rows]
    f1_label = f"{float(np.mean(f1)):.2f}" if f1 else "—"
    cosine_label = f"{float(np.mean(cosine)):.2f}" if cosine else "—"
    return f"{name}   F1 {f1_label}   cosine {cosine_label}   (n={len(rows)})"


def similarity_sizes(rows: list[SimilarityPoint]) -> list[float]:
    return [18.0 + 6.0 * np.sqrt(max(row.score.human_word_count, 1)) for row in rows]


def draw_similarity_scatter(
    ax,
    rows: list[SimilarityPoint],
    colors: dict[str, str],
) -> None:
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Token F1")
    ax.set_ylabel("Embedding cosine")
    sizes = similarity_sizes(rows)
    for name in unique_names([row.model for row in rows]):
        xs = [row.score.token_f1 for row in rows if row.model == name]
        ys = [row.score.embedding_cosine for row in rows if row.model == name]
        ss = [size for row, size in zip(rows, sizes) if row.model == name]
        ax.scatter(xs, ys, s=ss, c=colors[name], edgecolors="none", zorder=2)


def write_similarity_scatter(
    rows: list[SimilarityPoint],
    path: Path,
    colors: dict[str, str],
) -> None:
    names = unique_names([row.model for row in rows])
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    draw_similarity_scatter(ax, rows, colors)
    ax.set_title(similarity_caption(rows, names[0] if names else "Similarity"))
    save_figure(fig, path)


def write_similarity_strips(
    rows: list[SimilarityPoint],
    path: Path,
    colors: dict[str, str],
) -> None:
    names = unique_names([row.model for row in rows])
    groups = grouped(rows)
    fig, axes = plt.subplots(1, 2, figsize=(max(4.8, 1.4 * len(names) + 1.6), 3.2))
    for ax, attr, ylabel in (
        (axes[0], "token_f1", "Token F1"),
        (axes[1], "embedding_cosine", "Embedding cosine"),
    ):
        for index, name in enumerate(names):
            values = [getattr(row.score, attr) for row in groups[name]]
            span = 0.12
            offsets = (
                np.linspace(-span, span, len(values)) if len(values) > 1 else [0.0]
            )
            ax.scatter(
                [index + offset for offset in offsets],
                values,
                s=22,
                c=colors[name],
                edgecolors="none",
                zorder=2,
            )
            if values:
                mean = float(np.mean(values))
                ax.plot(
                    [index - 0.18, index + 0.18],
                    [mean, mean],
                    color="0.25",
                    lw=1.4,
                    solid_capstyle="butt",
                    zorder=3,
                )
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20)
        ax.set_ylabel(ylabel)
    fig.suptitle("Response similarity", y=1.02)
    save_figure(fig, path)


def write_similarity_summary(
    rows: list[SimilarityPoint],
    path: Path,
    colors: dict[str, str],
) -> None:
    names = unique_names([row.model for row in rows])
    if len(names) <= 1:
        write_similarity_scatter(rows, path, colors)
        return
    write_similarity_strips(rows, path, colors)


def latency_caption(rows: list[LatencyPoint]) -> str:
    parts = []
    for eval_type in unique_names([row.eval_type for row in rows]):
        values = [row.latency_ms for row in rows if row.eval_type == eval_type]
        mean = f"{float(np.mean(values)):.0f}ms" if values else "—"
        parts.append(f"{eval_type} {mean} (n={len(values)})")
    return "Response latency   " + "   ".join(parts)


def write_latency_strips(
    rows: list[LatencyPoint], path: Path, colors: dict[str, str]
) -> None:
    names = unique_names([row.model for row in rows])
    eval_types = unique_names([row.eval_type for row in rows])
    offsets = (
        np.linspace(-0.16, 0.16, len(eval_types))
        if len(eval_types) > 1
        else np.array([0.0])
    )
    fig, ax = plt.subplots(figsize=(max(4.0, 1.4 * len(names) + 1.6), 3.2))
    ymax = max((row.latency_ms for row in rows), default=0.0)
    for eval_index, eval_type in enumerate(eval_types):
        for model_index, name in enumerate(names):
            values = [
                row.latency_ms
                for row in rows
                if row.model == name and row.eval_type == eval_type
            ]
            if not values:
                continue
            span = 0.08
            jitter = (
                np.linspace(-span, span, len(values)) if len(values) > 1 else [0.0]
            )
            ax.scatter(
                [model_index + offsets[eval_index] + offset for offset in jitter],
                values,
                s=22,
                c=colors[name],
                edgecolors="none",
                zorder=2,
            )
            mean = float(np.mean(values))
            ax.plot(
                [
                    model_index + offsets[eval_index] - 0.12,
                    model_index + offsets[eval_index] + 0.12,
                ],
                [mean, mean],
                color="0.25",
                lw=1.4,
                solid_capstyle="butt",
                zorder=3,
            )
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0.0, ymax * 1.12 if ymax > 0 else 1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("Latency (ms)")
    ax.set_title(latency_caption(rows))
    save_figure(fig, path)


def write_metric_figures(
    run_dir: Path,
    eval_type: str,
    filename: str,
    rows: list,
    write_summary: Callable,
    write_detail: Callable,
) -> list[Path]:
    if not rows:
        return []
    written = [run_dir / filename]
    write_summary(rows, written[0])
    eval_dir = run_dir / eval_type
    stale = eval_dir / filename
    if stale.is_file():
        stale.unlink()
    if not eval_dir.is_dir():
        return written
    for name, group in grouped(rows).items():
        path = eval_dir / name / filename
        write_detail(group, path)
        written.append(path)
    return written


def write_run_dir_plots(
    run_dir: Path,
    artifacts: list[RunArtifact] | None = None,
) -> list[Path]:
    if artifacts is None:
        artifacts = load_run_artifacts(run_dir)
    pause, length, recall, similarity = plot_rows(artifacts)
    latency = latency_points(length, recall)
    colors = model_colors(
        [row.model for row in pause]
        + [row.model for row in length]
        + [row.model for row in recall]
        + [row.model for row in similarity]
    )
    written: list[Path] = []
    with plt.rc_context(STYLE):
        if pause:
            lim = pause_limit(pause)
            written.extend(
                write_metric_figures(
                    run_dir,
                    "pause-recognition",
                    PAUSE_NAME,
                    pause,
                    write_pause_bars,
                    lambda group, path: write_pause_figure(group, path, colors, lim),
                )
            )
        if length:
            duration_lim, word_lim = length_limits(length)
            written.extend(
                write_metric_figures(
                    run_dir,
                    "response-length",
                    LENGTH_NAME,
                    length,
                    lambda rows, path: write_length_grid(
                        rows, path, colors, duration_lim, word_lim
                    ),
                    lambda group, path: write_length_figure(
                        group, path, colors, duration_lim, word_lim
                    ),
                )
            )
            if len(unique_names([row.model for row in length])) > 1:
                average = run_dir / LENGTH_AVERAGE_NAME
                write_length_average(length, average, colors)
                written.append(average)
        if recall:
            root = run_dir / RECALL_NAME
            write_recall_figure(recall, root, colors)
            written.append(root)
            for eval_type in RECALL_EVAL_TYPES:
                typed = [row for row in recall if row.eval_type == eval_type]
                stale = run_dir / eval_type / RECALL_NAME
                if stale.is_file():
                    stale.unlink()
                if not typed:
                    continue
                for name, group in grouped(typed).items():
                    path = run_dir / eval_type / name / RECALL_NAME
                    write_recall_figure(group, path, colors)
                    written.append(path)
        if similarity:
            written.extend(
                write_metric_figures(
                    run_dir,
                    "response-length",
                    SIMILARITY_NAME,
                    similarity,
                    lambda rows, path: write_similarity_summary(rows, path, colors),
                    lambda group, path: write_similarity_scatter(group, path, colors),
                )
            )
        if latency:
            root = run_dir / LATENCY_NAME
            write_latency_strips(latency, root, colors)
            written.append(root)
    return written
