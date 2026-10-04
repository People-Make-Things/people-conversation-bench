"""Does removing voiced-but-unworded assistant history shorten the warm-up?

Reads `dose_probe.json`, the report from
`models/personaplex/deploy/dose.py::dose`: one point per distinct duplex
history in example_1, three prefill arms, several seeds per cell.

- `control` primes the unmasked history.
- `mask_unworded` zeroes the voiced history frames no stored word covers, which
  is the quantity that correlates with the warm-up at spearman +0.67.
- `mask_worded` zeroes the same number of voiced frames, spread over frames a
  word does cover. It removes the same amount of voiced audio and chops the
  history in the same way, so it separates the pairing from the audio.

Prefill is teacher-forced and deterministic, so within a cell the only thing the
seed changes is the continuation, which is where the outcome is measured. The
uncertainty is a cluster bootstrap over histories, because seeds within a history
share one prefill state and are not independent draws of the condition.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import DOSE as OUT  # noqa: E402
FRAME_RATE = 12.5
ARMS = ("control", "mask_unworded", "mask_worded")
BOOTSTRAP = 20000
RNG = np.random.default_rng(11)


def first_text_sec(arm: dict, continuation_sec: float) -> float:
    """Censored at the continuation length, which is what the drain would give."""
    frame = arm["continuation"]["first_text_frame"]
    return continuation_sec if frame is None else frame / FRAME_RATE


def load(path: Path) -> tuple[list[dict], float]:
    report = json.loads(path.read_text())
    continuation_sec = report["continuation_sec"]
    rows = []
    for arm in report["arms"]:
        continuation = arm["continuation"]
        rank = np.array(arm["rank"], dtype=float)
        rows.append(
            {
                "point_id": arm["point_id"],
                "arm": arm["arm"],
                "seed": arm["seed"],
                "history_sec": arm["history_sec"],
                "masked_sec": arm["masked_sec"],
                "frames": arm["frame_count"],
                "realtime_factor": arm["realtime_factor"],
                "median_rank": float(np.median(rank)) if rank.size else float("nan"),
                "rank_p90": float(np.percentile(rank, 90)) if rank.size else float("nan"),
                "first_text_sec": first_text_sec(arm, continuation_sec),
                "reached_text": continuation["first_text_frame"] is not None,
                "text_tokens": continuation["text_tokens"],
                "gated_share": continuation["gated_frames"] / continuation["frames"],
                "pretext_gated_share": (
                    continuation["gated_frames_before_text"] / continuation["frames"]
                ),
            }
        )
    return rows, continuation_sec


def cell(rows: list[dict], point_id: str, arm: str, key: str) -> np.ndarray:
    return np.array(
        [row[key] for row in rows if row["point_id"] == point_id and row["arm"] == arm],
        dtype=float,
    )


def cluster_bootstrap(per_point: dict[str, np.ndarray]) -> tuple[float, float, float]:
    """Mean paired difference with a 95% interval, resampling whole histories.

    Seeds inside one history share a prefill state, so resampling seeds would
    treat a repeated measurement as a new condition and understate the interval.
    The mean rather than the median because the outcome is censored at the
    continuation length in a third of the runs: with the censoring identical
    across arms the mean is a restricted mean difference, while a median of
    per-history medians collapses to zero whenever both arms are censored.
    """
    keys = list(per_point)
    observed = float(np.mean(np.concatenate([per_point[key] for key in keys])))
    draws = np.empty(BOOTSTRAP)
    for index in range(BOOTSTRAP):
        picked = RNG.choice(len(keys), size=len(keys), replace=True)
        draws[index] = np.mean(np.concatenate([per_point[keys[pick]] for pick in picked]))
    return observed, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def sign_flip_p(per_history: np.ndarray) -> float:
    """Exact two-sided randomization p over every relabelling of the arms.

    With seven paired histories the whole permutation set is 128 sign flips, and
    unlike the sign test this one uses how large each history's effect was, which
    is the difference between p=0.125 and a usable statement at this n.
    """
    observed = abs(per_history.mean())
    signs = np.array(list(itertools.product((1, -1), repeat=per_history.size)))
    return float(np.mean(np.abs((signs * per_history).mean(axis=1)) >= observed - 1e-12))


def sign_test(differences: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact sign test on one paired difference per history."""
    positive = int((differences > 0).sum())
    negative = int((differences < 0).sum())
    total = positive + negative
    if total == 0:
        return positive, negative, 1.0
    counts = np.arange(total + 1)
    weights = (
        np.array([math.comb(total, count) for count in counts], dtype=float) / 2**total
    )
    extreme = min(positive, negative)
    tail = float(weights[counts <= extreme].sum() + weights[counts >= total - extreme].sum())
    return positive, negative, min(1.0, tail)


def report(rows: list[dict], continuation_sec: float) -> None:
    points = sorted({row["point_id"] for row in rows})

    print(f"{len(rows)} runs, {len(points)} distinct histories, "
          f"{len({row['seed'] for row in rows})} seeds, "
          f"continuation {continuation_sec:g} s\n")

    print("Health")
    frames = {row["point_id"]: set() for row in rows}
    for row in rows:
        frames[row["point_id"]].add(row["frames"])
    print(f"  prefill frame count identical across arms per history: "
          f"{all(len(value) == 1 for value in frames.values())}")
    print(f"  realtime factor min {min(row['realtime_factor'] for row in rows):.2f}")
    for arm in ARMS:
        median_rank = np.median([row["median_rank"] for row in rows if row["arm"] == arm])
        p90 = np.median([row["rank_p90"] for row in rows if row["arm"] == arm])
        print(f"  {arm:14s} forced-text rank median {median_rank:.1f}, p90 {p90:.1f}")

    print("\nTime to first text token, seconds (censored at the continuation length)")
    header = "  ".join(f"{arm:>22s}" for arm in ARMS)
    print(f"  {'history':10s} {'hist_s':>7} {'dose_s':>7}  {header}")
    for point_id in points:
        line = f"  {point_id.split('_')[-1]:10s}"
        line += f" {cell(rows, point_id, 'control', 'history_sec')[0]:7.2f}"
        line += f" {cell(rows, point_id, 'mask_unworded', 'masked_sec')[0]:7.2f} "
        for arm in ARMS:
            values = cell(rows, point_id, arm, "first_text_sec")
            reached = int(cell(rows, point_id, arm, "reached_text").sum())
            line += (
                f"  {np.median(values):6.2f} "
                f"[{values.min():5.2f},{values.max():5.2f}] {reached}/{values.size}"
            )
        print(line)

    print("\nArm means over all runs")
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        reached = sum(row["reached_text"] for row in subset)
        print(
            f"  {arm:14s} first text median "
            f"{np.median([row['first_text_sec'] for row in subset]):5.2f} s, "
            f"mean {np.mean([row['first_text_sec'] for row in subset]):5.2f} s, "
            f"reached {reached}/{len(subset)}, "
            f"gated {np.mean([row['gated_share'] for row in subset]):.3f}, "
            f"pre-text gated {np.mean([row['pretext_gated_share'] for row in subset]):.3f}"
        )

    for key, name in (
        ("first_text_sec", "time to first text (s), restricted mean"),
        ("reached_text", "reached text inside the continuation"),
        ("pretext_gated_share", "share of continuation above the gate before first text"),
        ("gated_share", "share of continuation above the gate"),
    ):
        print(f"\nPaired effect on {name}, versus control")
        for arm in ARMS[1:]:
            per_point = {}
            for point_id in points:
                # Seeds are aligned across arms, so the difference is paired on
                # the seed as well as the history.
                per_point[point_id] = cell(rows, point_id, arm, key) - cell(
                    rows, point_id, "control", key
                )
            observed, low, high = cluster_bootstrap(per_point)
            per_history = np.array([np.mean(value) for value in per_point.values()])
            positive, negative, pvalue = sign_test(per_history)
            print(
                f"  {arm:14s} mean difference {observed:+.3f} "
                f"(95% CI {low:+.3f} to {high:+.3f}), "
                f"histories worse/better {positive}/{negative}, sign test p={pvalue:.3f}, "
                f"randomization p={sign_flip_p(per_history):.3f}"
            )
            print(
                "                 per history "
                + " ".join(f"{value:+.2f}" for value in per_history)
            )


def plot(rows: list[dict], continuation_sec: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = sorted({row["point_id"] for row in rows})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(points)))

    for index, point_id in enumerate(points):
        history_sec = cell(rows, point_id, "control", "history_sec")[0]
        for axis, key in ((axes[0], "first_text_sec"), (axes[1], "pretext_gated_share")):
            medians = [np.median(cell(rows, point_id, arm, key)) for arm in ARMS]
            axis.plot(
                range(len(ARMS)),
                medians,
                marker="o",
                color=colors[index],
                label=f"{history_sec:.0f} s history",
            )
            for offset, arm in enumerate(ARMS):
                values = cell(rows, point_id, arm, key)
                axis.scatter(
                    np.full(values.shape, offset) + RNG.uniform(-0.08, 0.08, values.shape),
                    values,
                    s=9,
                    alpha=0.45,
                    color=colors[index],
                )

    axes[0].axhline(continuation_sec, color="0.6", ls="--", lw=1)
    axes[0].set_ylabel("time to first text token (s)")
    axes[0].set_title(f"warm-up per arm (dashed: censored at {continuation_sec:g} s)")
    axes[1].set_ylabel("share above gate before first text")
    axes[1].set_title("wordless phonation before the first token")
    for axis in axes[:2]:
        axis.set_xticks(range(len(ARMS)))
        axis.set_xticklabels(ARMS, rotation=12)
        axis.grid(alpha=0.3)
    axes[0].legend(fontsize=7)

    for arm, offset in zip(ARMS[1:], (-0.18, 0.18), strict=True):
        differences = [
            np.mean(
                cell(rows, point_id, arm, "first_text_sec")
                - cell(rows, point_id, "control", "first_text_sec")
            )
            for point_id in points
        ]
        axes[2].scatter(
            np.full(len(differences), offset) + RNG.uniform(-0.05, 0.05, len(differences)),
            differences,
            s=42,
            color="#1f77b4" if offset < 0 else "#d62728",
            label=arm,
        )
        axes[2].plot([offset - 0.1, offset + 0.1], [np.median(differences)] * 2, color="k")
    axes[2].axhline(0, color="0.4", lw=1)
    axes[2].set_xlim(-0.5, 0.5)
    axes[2].set_xticks([])
    axes[2].set_ylabel("per-history change in first text (s)")
    axes[2].set_title("paired effect, restricted mean per history")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(OUT / "dose.png", dpi=150)
    print(f"\nwrote {(OUT / 'dose.png').resolve()}")


def main() -> None:
    rows, continuation_sec = load(OUT / "dose_probe.json")
    report(rows, continuation_sec)
    plot(rows, continuation_sec)


if __name__ == "__main__":
    main()
