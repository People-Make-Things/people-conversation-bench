"""What does the wordless phonation track?

A pipeline defect correlates with something mechanical (history length, prefill
cost, position in the run, container age). Natural backchannelling correlates
with conversational context. Also: where does the phonation sit relative to the
scored pause window, and how do span durations compare with human backchannels.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import CELL_SEC, OUT, ROOT, analyze, load_cache, read, runs  # noqa: E402
from bench.speech import observed_input_time  # noqa: E402
from bench.transcribe import load_whisper_model  # noqa: E402

RESULTS = ROOT / "data/results/personaplex/baseline_v2"
POINTS = ROOT / "data/processed/example_1/points"


def spearman(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    keep = np.isfinite(left) & np.isfinite(right)
    left, right = left[keep], right[keep]
    if left.size < 4:
        return float("nan"), int(left.size)
    rank = lambda values: np.argsort(np.argsort(values)).astype(float)
    a, b = rank(left) - rank(left).mean(), rank(right) - rank(right).mean()
    scale = np.linalg.norm(a) * np.linalg.norm(b)
    return (float(np.dot(a, b) / scale) if scale > 0 else float("nan")), int(left.size)


def main() -> None:
    report = json.loads((OUT / "cohorts.json").read_text())
    character = json.loads((OUT / "character.json").read_text())
    cache = load_cache()
    model = load_whisper_model()

    print("Wordless above-gate span durations (s)")
    print(f"{'cohort':30s} {'n':>4} {'p50':>6} {'p75':>6} {'p90':>6} {'max':>6} {'>1.0s':>7} {'>2.0s':>7}")
    for name in (
        "model_pause:wordless",
        "model_eot:wordless",
        "model_pause_prefix:wordless",
        "human_eot:wordless",
        "human_full:wordless",
    ):
        values = np.array([s["duration_sec"] for s in character[name]])
        if values.size == 0:
            continue
        print(
            f"{name:30s} {values.size:4d} {np.percentile(values, 50):6.2f} "
            f"{np.percentile(values, 75):6.2f} {np.percentile(values, 90):6.2f} "
            f"{values.max():6.2f} {int((values > 1.0).sum()):7d} "
            f"{int((values > 2.0).sum()):7d}"
        )

    rows: list[dict] = []
    for row in report["model_pause"]:
        _, point_name, rollout_name = row["key"].split("/")
        trace_dir = RESULTS / point_name / rollout_name
        trace = json.loads((trace_dir / "trace.json").read_text())
        meta = json.loads(
            (POINTS / point_name.split("_")[-1] / "point.json").read_text()
        )
        pcm, rate = read(Path(row["path"]))
        analysis = analyze(model, row["key"], pcm, rate, cache)
        phonation = analysis.phonation()
        session = trace.get("session", {})
        events = [
            type("E", (), {"kind": e["kind"], "at_sec": e["at_sec"], "data": e.get("data", {})})()
            for e in trace["events"]
        ]
        checkpoint = observed_input_time(events, trace["checkpoint_sec"])
        window_end = observed_input_time(events, trace["window_end_sec"])
        blocks = trace["played_audio"]
        starts = np.array([b["start_sec"] for b in blocks])
        length = min(starts.shape[0], phonation.shape[0])
        in_window = phonation[:length] & (starts[:length] < window_end) & (
            starts[:length] >= checkpoint
        )
        before = phonation[:length] & (starts[:length] < checkpoint)
        after = phonation[:length] & (starts[:length] >= window_end)
        rows.append(
            {
                "key": row["key"],
                "point": point_name,
                "rollout_index": int(rollout_name.split("_")[-1]),
                "history_end_sec": meta["rollout"]["history_end_sec"],
                "pause_duration_sec": trace["window_end_sec"] - trace["checkpoint_sec"],
                "live_turn_sec": trace["window_end_sec"],
                "realtime_factor": session.get("realtime_factor"),
                "prefill_sec": session.get("prefill_sec"),
                "duration_sec": row["duration_sec"],
                "phonation_sec": row["phonation_sec"],
                "phonation_fraction": row["phonation_sec"] / row["duration_sec"],
                "lexical_sec": row["labeled_sec"],
                "in_window_sec": float(in_window.sum()) * CELL_SEC,
                "before_checkpoint_sec": float(before.sum()) * CELL_SEC,
                "after_window_sec": float(after.sum()) * CELL_SEC,
            }
        )
    (OUT / "correlate.json").write_text(json.dumps(rows))

    print("\nSession fields present in trace.json:")
    print(" ", sorted({k for r in report["model_pause"] for k in ()} ) or
          sorted(json.loads((RESULTS / rows[0]["point"] / f"rollout_{rows[0]['rollout_index']:03d}" / "trace.json").read_text()).get("session", {})))

    print("\nSpearman correlation of per-rollout wordless-phonation fraction with:")
    fraction = np.array([r["phonation_fraction"] for r in rows])
    for field in (
        "history_end_sec",
        "pause_duration_sec",
        "live_turn_sec",
        "duration_sec",
        "rollout_index",
        "realtime_factor",
        "prefill_sec",
    ):
        values = np.array(
            [float(r[field]) if r[field] is not None else np.nan for r in rows]
        )
        rho, n = spearman(values, fraction)
        print(f"  {field:24s} rho={rho:+.3f}  (n={n})")

    print("\nWhere the wordless phonation sits, summed over 57 pause rollouts:")
    for field, label in (
        ("before_checkpoint_sec", "before checkpoint"),
        ("in_window_sec", "inside the scored pause window"),
        ("after_window_sec", "after window_end"),
    ):
        print(f"  {label:34s} {sum(r[field] for r in rows):7.1f} s")

    print("\nPer-point wordless phonation fraction (mean of 3 rollouts):")
    points = sorted({r["point"] for r in rows})
    for point in points:
        group = [r for r in rows if r["point"] == point]
        print(
            f"  {point:20s} hist={group[0]['history_end_sec']:6.2f}s "
            f"pause={group[0]['pause_duration_sec']:5.2f}s "
            f"phon_frac={np.mean([g['phonation_fraction'] for g in group]):.3f} "
            f"lex_s={np.mean([g['lexical_sec'] for g in group]):5.2f}"
        )


if __name__ == "__main__":
    main()
