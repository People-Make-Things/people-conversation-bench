"""Figures for the drain-length and prefill-conditioning experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import OUT  # noqa: E402
from experiments import REPORT_HORIZON_SEC  # noqa: E402

# baseline_v2's median rollout: where the old protocol stopped watching.
BASELINE_MEDIAN_SEC = 6.9

LABELS = {
    "exp1_drain15": "full history, aligned text\n(drain 15 s)",
    "exp2_padtext": "full history audio,\nPAD text (drain 15 s)",
    "exp2_hist4": "last 4 s of history\n(drain 15 s)",
    "exp2_hist0": "one frame of history\n(drain 15 s)",
    "baseline_v2": "baseline_v2\n(drain 3 s)",
}
COLORS = {
    "exp1_drain15": "#c1442e",
    "exp2_padtext": "#e8a33d",
    "exp2_hist4": "#2b7bba",
    "exp2_hist0": "#4a9c6d",
    "baseline_v2": "#5a5a5a",
}
ORDER = ["baseline_v2", "exp1_drain15", "exp2_padtext", "exp2_hist4", "exp2_hist0"]


def first_text_ecdf(axis, rows: list[dict], label: str, color: str) -> None:
    """Share of rollouts that had produced a text token by each elapsed time.

    Every rollout is observed past REPORT_HORIZON_SEC in the drain-15 runs, so
    over that range this is an uncensored CDF. baseline_v2 rollouts end sooner,
    so its curve is drawn only as far as its shortest rollout ran.
    """
    observed = np.array([row["observed_sec"] for row in rows])
    times = np.array(
        [row["first_text_sec"] for row in rows if row["first_text_sec"] is not None]
    )
    limit = min(float(observed.min()), REPORT_HORIZON_SEC + 1.0)
    grid = np.linspace(0, limit, 400)
    share = [(times <= point).sum() / len(rows) for point in grid]
    axis.step(grid, share, where="post", color=color, lw=2.0, label=f"{label} (n={len(rows)})")
    axis.plot([limit], [share[-1]], "o", color=color, ms=5)


def figure(report: dict, cohorts: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    for name in ORDER:
        rows = report.get(name)
        if rows:
            first_text_ecdf(
                axes[0], rows, LABELS[name].replace("\n", " "), COLORS[name]
            )
    axes[0].axvline(BASELINE_MEDIAN_SEC, color="#444444", ls="--", lw=1)
    axes[0].text(
        BASELINE_MEDIAN_SEC + 0.2,
        0.03,
        f"baseline_v2 median\nrollout ({BASELINE_MEDIAN_SEC} s)",
        fontsize=7.5,
        color="#444444",
    )
    axes[0].set_xlabel("seconds since the first model audio frame")
    axes[0].set_ylabel("share of rollouts that have emitted text")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(
        "Time to the model's first text token\n(dot = end of the observed window)",
        fontsize=10,
    )
    axes[0].legend(fontsize=7.5, loc="upper left")
    axes[0].grid(alpha=0.25)

    names = [name for name in ORDER if report.get(name)]
    positions = np.arange(len(names))
    ever = [
        np.mean([row["first_text_sec"] is not None for row in report[name]])
        for name in names
    ]
    gate = [np.mean([row["speech_ratio"] for row in report[name]]) for name in names]
    axes[1].barh(
        positions + 0.19,
        ever,
        height=0.36,
        color=[COLORS[name] for name in names],
        label="produced any text token",
    )
    axes[1].barh(
        positions - 0.19,
        gate,
        height=0.36,
        color="none",
        edgecolor=[COLORS[name] for name in names],
        hatch="///",
        label="share of the rollout above the speech gate",
    )
    for position, value in zip(positions, ever, strict=True):
        axes[1].text(value + 0.02, position + 0.19, f"{value:.2f}", va="center", fontsize=8)
    for position, value in zip(positions, gate, strict=True):
        axes[1].text(value + 0.02, position - 0.19, f"{value:.2f}", va="center", fontsize=8)
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels([LABELS[name] for name in names], fontsize=8)
    axes[1].set_xlim(0, 1.15)
    axes[1].set_title(
        "Does the text head engage at all, and\nhow much audio is it running under?",
        fontsize=10,
    )
    axes[1].legend(fontsize=7.5, loc="lower right")
    axes[1].grid(axis="x", alpha=0.25)

    cohort_names = [name for name in ORDER if cohorts.get(cohort_key(name))]
    positions = np.arange(len(cohort_names))
    shares = []
    for name in cohort_names:
        rows = cohorts[cohort_key(name)]
        gated = sum(row["gated_sec"] for row in rows)
        phonation = sum(row["phonation_sec"] for row in rows)
        shares.append(phonation / gated if gated else 0.0)
    axes[2].barh(positions, shares, color=[COLORS[name] for name in cohort_names])
    for position, value in zip(positions, shares, strict=True):
        axes[2].text(value + 0.015, position, f"{value:.3f}", va="center", fontsize=8)
    axes[2].axvline(0.5, color="#444444", ls="--", lw=1)
    axes[2].set_ylim(-1.1, len(cohort_names) - 0.4)
    axes[2].text(0.51, -0.95, "detector floor inside\ngenuine speech", fontsize=7, color="#444444")
    gpt = cohorts.get("model_pause_gpt")
    if gpt:
        gated = sum(row["gated_sec"] for row in gpt)
        value = sum(row["phonation_sec"] for row in gpt) / gated
        axes[2].axvline(value, color="#2b7bba", ls=":", lw=1.5)
        axes[2].text(value + 0.01, len(cohort_names) - 0.6, "GPT Realtime", fontsize=7, color="#2b7bba")
    axes[2].set_yticks(positions)
    axes[2].set_yticklabels([LABELS[name] for name in cohort_names], fontsize=8)
    axes[2].set_xlim(0, 1)
    axes[2].set_title(
        "Share of above-gate audio with no\ndecodable word (Whisper labels)",
        fontsize=10,
    )
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Is the silent text head a bounded warm-up, and does prefill conditioning cause it?",
        fontsize=12,
    )
    fig.tight_layout()
    path = OUT / "experiments.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def cohort_key(name: str) -> str:
    return "model_pause" if name == "baseline_v2" else name


def mechanism_figure(report: dict, gaps: list[dict]) -> Path:
    """Where the phonation sits relative to the first token, and what predicts it."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    names = [name for name in ORDER if report.get(name)]
    positions = np.arange(len(names))
    pre = [
        np.mean([row["gated_before_text_sec"] for row in report[name]]) for name in names
    ]
    post = [
        np.mean([row["gated_after_text_sec"] for row in report[name]]) for name in names
    ]
    axes[0].barh(positions, pre, height=0.6, color=[COLORS[name] for name in names],
                 label="text head still silent")
    axes[0].barh(positions, post, height=0.6, left=pre, color="none",
                 edgecolor=[COLORS[name] for name in names], hatch="///",
                 label="after the first text token")
    for position, before, after in zip(positions, pre, post, strict=True):
        axes[0].text(before + after + 0.1, position, f"{before:.1f} + {after:.1f} s",
                     va="center", fontsize=8)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([LABELS[name] for name in names], fontsize=8)
    axes[0].set_xlabel("seconds above the speech gate, per rollout")
    axes[0].set_xlim(0, max(b + a for b, a in zip(pre, post, strict=True)) * 1.35)
    axes[0].set_title(
        "Above-gate audio before and after\nthe text head engages", fontsize=10
    )
    axes[0].legend(fontsize=7.5, loc="lower right")
    axes[0].grid(axis="x", alpha=0.25)

    share = np.array([entry["unworded_share_of_voiced"] for entry in gaps])
    first = np.array([entry["median_first_text_sec"] for entry in gaps])
    history = np.array([entry["history_sec"] for entry in gaps])
    scatter = axes[1].scatter(share, first, c=history, cmap="viridis", s=70,
                              edgecolor="#333333", lw=0.5)
    fig.colorbar(scatter, ax=axes[1], label="history length (s)")
    fit = np.polyfit(share, first, 1)
    grid = np.linspace(share.min(), share.max(), 50)
    axes[1].plot(grid, np.polyval(fit, grid), color="#c1442e", ls="--", lw=1.2)
    axes[1].set_xlabel("share of voiced prefill history with no word over it")
    axes[1].set_ylabel("median time to first text token (s)")
    axes[1].set_title(
        f"Unworded prefill audio predicts the warm-up\n"
        f"(spearman +0.67 over {len(gaps)} points; history alone +0.35)",
        fontsize=10,
    )
    axes[1].grid(alpha=0.25)

    fig.suptitle(
        "The phonation is what the model does before its text head engages, "
        "and unworded prefill audio delays that",
        fontsize=11,
    )
    fig.tight_layout()
    path = OUT / "experiments_mechanism.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    report = json.loads((OUT / "experiments.json").read_text())
    cohorts = json.loads((OUT / "cohorts.json").read_text())
    gaps = json.loads((OUT / "prefill_gaps.json").read_text())
    print(figure(report, cohorts))
    print(mechanism_figure(report, gaps))


if __name__ == "__main__":
    main()
