"""What rank does PersonaPlex give the text token prefill forces on it?

Reads `rank_probe.json`, written by
`models/personaplex/deploy/rank_probe.py::probe`: a voice prompt x history
length factorial over three points, measured on the GPU with the real
`ServerState` and the real `run_duplex_prefill`. Each arm then continues on
silence frames the way the pause drain does, so a rank trajectory can be read
against a time-to-first-text.

The rank is the position of the forced token in the text head's own ranking for
that frame, which is the quantity upstream's `create_loss_report` puts in
`ranks_of_forced[:, 0]`. Rank 0 means the model would have emitted that token
itself. A rank that climbs across prefill would be the text head being driven
out of distribution by its own conditioning.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parent
FRAME_SEC = 0.08
PAD_TOKEN = 3
EPAD_TOKEN = 0
CLASSES = ("PAD", "EPAD", "word")
COLORS = {"PAD": "#4a9c6d", "EPAD": "#e8a33d", "word": "#c1442e"}


def token_class(forced: np.ndarray) -> np.ndarray:
    classes = np.full(forced.shape, "word", dtype=object)
    classes[forced == PAD_TOKEN] = "PAD"
    classes[forced == EPAD_TOKEN] = "EPAD"
    return classes


def arm_label(arm: dict) -> str:
    voice = arm["voice_prompt"] or "no voice prompt"
    return f"{arm['point_id'][-6:]} {arm['history']:>4} history, {voice}"


def rank_table(arms: list[dict]) -> None:
    print("Rank of the forced text token, by what was forced")
    print(
        f"  {'point':7s} {'hist':>5} {'voice':>9} {'n':>4} "
        f"{'PAD med/p90':>13} {'EPAD med/p90':>13} {'word med/p90':>13} "
        f"{'top1=PAD|PAD':>13} {'rank>100':>9}"
    )
    for arm in arms:
        rank = np.array(arm["rank"])
        forced = np.array(arm["forced"])
        top = np.array(arm["top"])
        classes = token_class(forced)
        cells = ""
        for name in CLASSES:
            mask = classes == name
            cells += (
                f"{np.median(rank[mask]):6.0f}/{np.percentile(rank[mask], 90):6.0f} "
                if mask.any()
                else f"{'-':>13} "
            )
        pad = forced == PAD_TOKEN
        print(
            f"  {arm['point_id'][-6:]:7s} {arm['history']:>5} "
            f"{(arm['voice_prompt'] or 'none'):>9} {rank.shape[0]:4d} {cells}"
            f"{np.mean(top[pad] == PAD_TOKEN):13.3f} {int((rank > 100).sum()):9d}"
        )


def trajectory_table(arms: list[dict]) -> None:
    print("\nMedian and mean rank over fifths of prefill (no climb = in distribution)")
    print(f"  {'point':7s} {'hist':>5} {'voice':>9} " + " ".join(f"{k + 1:>13}" for k in range(5)))
    for arm in arms:
        rank = np.array(arm["rank"])
        fifths = np.array_split(rank, 5)
        print(
            f"  {arm['point_id'][-6:]:7s} {arm['history']:>5} "
            f"{(arm['voice_prompt'] or 'none'):>9} "
            + " ".join(f"{np.median(part):5.0f}/{part.mean():7.1f}" for part in fifths)
        )


def continuation_table(arms: list[dict]) -> None:
    print("\nContinuation on silence after prefill, the same 15 s the drain streams")
    print(
        f"  {'point':7s} {'hist':>5} {'voice':>9} {'forced PAD':>11} "
        f"{'end prob':>9} {'first text':>11} {'tokens':>7} {'gate%':>7} {'gate% pre-text':>15}"
    )
    for arm in arms:
        forced = np.array(arm["forced"])
        prob = np.array(arm["prob"])
        continuation = arm["continuation"]
        first = continuation["first_text_frame"]
        print(
            f"  {arm['point_id'][-6:]:7s} {arm['history']:>5} "
            f"{(arm['voice_prompt'] or 'none'):>9} "
            f"{np.mean(forced == PAD_TOKEN):11.2f} {prob[-12:].mean():9.3f} "
            f"{(f'{first * FRAME_SEC:.2f} s' if first is not None else 'none'):>11} "
            f"{continuation['text_tokens']:7d} "
            f"{continuation['gated_frames'] / continuation['frames'] * 100:7.1f} "
            f"{continuation['gated_frames_before_text'] / continuation['frames'] * 100:15.1f}"
        )


def figure(report: dict) -> Path:
    arms = report["arms"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    pooled = {name: [] for name in CLASSES}
    for arm in arms:
        if arm["history"] != "full":
            continue
        classes = token_class(np.array(arm["forced"]))
        rank = np.array(arm["rank"])
        for name in CLASSES:
            pooled[name].append(rank[classes == name])
    for name in CLASSES:
        values = np.concatenate(pooled[name])
        grid = np.unique(np.concatenate([[0], values]))
        share = [(values <= point).mean() for point in grid]
        axes[0].step(
            grid + 1,
            share,
            where="post",
            color=COLORS[name],
            lw=2.0,
            label=f"{name} (n={values.shape[0]})",
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("rank of the forced token, plus one")
    axes[0].set_ylabel("share of frames at or below")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_title(
        "The forced text is what the model would\nhave said (full-history arms)", fontsize=10
    )
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].grid(alpha=0.25)

    longest = max(arms, key=lambda arm: arm["frame_count"])
    for arm in arms:
        if arm["point_id"] != longest["point_id"] or arm["history"] != "full":
            continue
        rank = np.array(arm["rank"])
        seconds = np.arange(rank.shape[0]) * FRAME_SEC
        colour = "#c1442e" if arm["voice_prompt"] else "#2b7bba"
        axes[1].plot(
            seconds,
            rank + 1,
            ".",
            ms=3,
            color=colour,
            alpha=0.6,
            label=arm["voice_prompt"] or "no voice prompt",
        )
    axes[1].set_yscale("log")
    axes[1].set_xlabel(f"seconds into prefill ({longest['point_id'][-6:]}, {longest['history']})")
    axes[1].set_ylabel("rank of the forced token, plus one")
    axes[1].set_title(
        "The rank does not climb: only the first frame\nand one word onset leave the floor",
        fontsize=10,
    )
    axes[1].legend(fontsize=8, loc="upper center")
    axes[1].grid(alpha=0.25)

    labels = []
    values = []
    colours = []
    for arm in sorted(arms, key=lambda arm: (arm["point_id"], arm["history"])):
        first = arm["continuation"]["first_text_frame"]
        labels.append(arm_label(arm))
        values.append(
            report["continuation_sec"] if first is None else first * FRAME_SEC
        )
        colours.append("#c1442e" if arm["voice_prompt"] else "#2b7bba")
    positions = np.arange(len(labels))
    axes[2].barh(positions, values, color=colours)
    for position, arm, value in zip(
        positions,
        sorted(arms, key=lambda arm: (arm["point_id"], arm["history"])),
        values,
        strict=True,
    ):
        censored = arm["continuation"]["first_text_frame"] is None
        axes[2].text(
            value + 0.2,
            position,
            "no text in 15 s" if censored else f"{value:.1f} s",
            va="center",
            fontsize=7.5,
        )
    axes[2].set_yticks(positions)
    axes[2].set_yticklabels(labels, fontsize=7.5)
    axes[2].set_xlim(0, report["continuation_sec"] * 1.35)
    axes[2].set_xlabel("seconds to the first text token after prefill")
    axes[2].set_title(
        "Restoring the voice prompt (red) does not\nshorten the warm-up", fontsize=10
    )
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Forced-text rank over prefill: the text head is in distribution, "
        "and the voice prompt is not the difference",
        fontsize=12,
    )
    fig.tight_layout()
    path = OUT / "rank_probe.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    report = json.loads((OUT / "rank_probe.json").read_text())
    arms = report["arms"]
    print(
        f"personaplex {report['personaplex_commit'][:8]}, seed {report['seed']}, "
        f"{len(arms)} arms, {report['continuation_sec']:.0f} s continuation"
    )
    rank_table(arms)
    trajectory_table(arms)
    continuation_table(arms)
    print(f"\n{figure(report)}")


if __name__ == "__main__":
    main()
