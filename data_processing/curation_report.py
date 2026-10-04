"""Figures summarizing what curation kept and what it cut, and why."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_NAME = "curation.png"
JUDGE_FIGURE_NAME = "curation_judge.png"
KEPT_COLOR = "#54a24b"
CUT_COLOR = "#e45756"
OUTRANKED_COLOR = "#f58518"
SPLIT_COLORS = {"pause_start": "#4c78a8", "end_of_turn": "#b279a2"}


def load_events(curation_dir: Path) -> list[dict]:
    events = []
    for path in sorted(curation_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for event in payload["events"]:
            events.append({**event, "source_id": payload["source_id"]})
    return events


def reason_axis(ax, events: list[dict]) -> None:
    counts = Counter(event["reason"] for event in events)
    reasons = [reason for reason, _ in counts.most_common()][::-1]
    colors = [
        KEPT_COLOR
        if reason == "selected"
        else OUTRANKED_COLOR
        if reason == "outranked"
        else CUT_COLOR
        for reason in reasons
    ]
    ax.barh(reasons, [counts[reason] for reason in reasons], color=colors)
    ax.set_title("Decision by reason")
    ax.set_xlabel("split events")


def feature_axis(ax, events: list[dict], feature: str, title: str, unit: str) -> None:
    kept = [
        event["features"][feature]
        for event in events
        if event["keep"] and event["features"].get(feature) is not None
    ]
    cut = [
        event["features"][feature]
        for event in events
        if not event["keep"] and event["features"].get(feature) is not None
    ]
    # Kept counts are orders of magnitude below cut counts, so overlaid bars on
    # a log scale are the only way both distributions stay visible.
    ax.hist(cut, bins=20, color=CUT_COLOR, alpha=0.7, label="cut")
    ax.hist(kept, bins=20, color=KEPT_COLOR, alpha=0.9, label="kept")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(unit)
    ax.set_ylabel("split events (log)")
    ax.legend(frameon=False)


def score_axis(ax, events: list[dict]) -> None:
    for split_type, color in (("pause_start", "#4c78a8"), ("end_of_turn", "#b279a2")):
        scores = [
            event["score"]
            for event in events
            if event["split_type"] == split_type and event["score"] is not None
        ]
        if scores:
            ax.hist(scores, bins=20, alpha=0.6, color=color, label=split_type)
    ax.set_title("Scores of gate survivors (kept + outranked)")
    ax.set_xlabel("score")
    ax.set_ylabel("split events")
    ax.legend(frameon=False)


def judge_outcome(event: dict) -> str:
    """Why the judge kept or killed a gate survivor, as a stable label."""
    if event["keep"]:
        return "selected"
    if event["reason"] == "outranked":
        return "outranked"
    judge = event["judge"]
    if not judge["coherent"]:
        return "incoherent"
    if event["split_type"] == "pause_start":
        if judge["turn_handoff"]:
            return "turn_handoff"
        return "low_temptation"
    if not judge["genuine_handoff"]:
        return "interrupted_fragment"
    if not judge["responsive_reference"]:
        return "unresponsive_reference"
    return "low_difficulty"


def judge_outcome_axis(ax, events: list[dict]) -> None:
    outcomes = sorted(
        {judge_outcome(event) for event in events},
        key=lambda outcome: -sum(judge_outcome(e) == outcome for e in events),
    )
    positions = range(len(outcomes))
    width = 0.4
    for offset, split_type in ((-width / 2, "pause_start"), (width / 2, "end_of_turn")):
        counts = [
            sum(
                1
                for event in events
                if event["split_type"] == split_type
                and judge_outcome(event) == outcome
            )
            for outcome in outcomes
        ]
        ax.bar(
            [p + offset for p in positions],
            counts,
            width,
            color=SPLIT_COLORS[split_type],
            label=split_type,
        )
    ax.set_xticks(list(positions))
    ax.set_xticklabels(outcomes, rotation=30, ha="right")
    ax.set_title("Judge outcome for gate survivors")
    ax.set_ylabel("split events")
    ax.legend(frameon=False)


def judge_scatter_axis(ax, events: list[dict]) -> None:
    for keep, color, label in ((False, CUT_COLOR, "cut"), (True, KEPT_COLOR, "kept")):
        sub = [event for event in events if event["keep"] == keep]
        ax.scatter(
            [event["score"] for event in sub],
            [event["judge"]["score"] for event in sub],
            s=12,
            alpha=0.4,
            color=color,
            label=label,
        )
    ax.set_title("Heuristic score vs judge score")
    ax.set_xlabel("heuristic score")
    ax.set_ylabel("judge score")
    ax.legend(frameon=False)


def write_judge_figures(curation_dir: Path) -> Path | None:
    events = [event for event in load_events(curation_dir) if event.get("judge")]
    if not events:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    judge_outcome_axis(axes[0], events)
    judge_scatter_axis(axes[1], events)
    kept = sum(1 for event in events if event["keep"])
    fig.suptitle(
        f"LLM judge over {len(events)} gate survivors: {kept} selected "
        f"({events[0]['judge']['model']})"
    )
    fig.tight_layout()
    figure_path = curation_dir / JUDGE_FIGURE_NAME
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path


def write_curation_figures(curation_dir: Path) -> Path:
    events = load_events(curation_dir)
    if not events:
        raise ValueError(f"no curation files under {curation_dir}")
    kept = sum(1 for event in events if event["keep"])
    sources = len({event["source_id"] for event in events})

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    reason_axis(axes[0][0], events)
    feature_axis(
        axes[0][1],
        [event for event in events if event["split_type"] == "pause_start"],
        "pause_sec",
        "Pause duration, kept vs cut",
        "seconds",
    )
    feature_axis(
        axes[1][0],
        [event for event in events if event["split_type"] == "end_of_turn"],
        "reference_word_count",
        "EOT reference length, kept vs cut",
        "reference words",
    )
    score_axis(axes[1][1], events)
    fig.suptitle(
        f"Curation: kept {kept} of {len(events)} split events "
        f"from {sources} source(s)"
    )
    fig.tight_layout()
    figure_path = curation_dir / FIGURE_NAME
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)
    return figure_path
