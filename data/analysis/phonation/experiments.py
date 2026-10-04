"""Bounded warm-up or unbounded degenerate state, and does prefill cause it?

Four live runs, all pause points, all with the drain raised to 15 s so a rollout
lasts as long as an EOT one:

  exp1_drain15   full history, aligned assistant text. Differs from baseline_v2
                 only in rollout length, so it says whether the silent text head
                 is a startup transient the old 6.9 s rollouts ended inside.
  exp2_padtext   the same history audio with no word stream, so prefill
                 teacher-forces all PAD.
  exp2_hist4     the last 4 s of history, aligned text kept.
  exp2_hist0     one 80 ms frame of history: the persona prompt and nothing else.

padtext and hist4 move one thing each off exp1_drain15, so "the forced text is
malformed" and "a long history degrades the state" are separable; hist0 is the
anchor with no conditioning at all.

Everything here comes from trace.json and response.wav. The model's own text
stream is the primary measurement and owes nothing to whisper or to any RMS
threshold. Rollouts all run past 15 s of model audio, so the time-to-first-text
distribution is uncensored over the window it is reported on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import OUT, ROOT, read  # noqa: E402
from bench.audio_metrics import speech_ratio  # noqa: E402
from bench.speech import PlayedAudio, speech_intervals  # noqa: E402

RESULTS = ROOT / "data/results/personaplex"
CONDITIONS = {
    "exp1_drain15": "full history, aligned text",
    "exp2_padtext": "full history audio, PAD text",
    "exp2_hist4": "last 4 s of history",
    "exp2_hist0": "one frame of history",
    "baseline_v2": "full history, drain 3 s",
}
REPORT_HORIZON_SEC = 15.0
# baseline_v2's shortest rollout, so every condition is observed over this much.
MATCHED_WINDOW_SEC = 3.7


def gated_sec_split(trace: dict, origin: float, first_text_sec: float | None) -> tuple[float, float]:
    """Seconds above the speech gate before and after the first text token.

    On the played-audio clock, which is the clock model_text events carry and
    the one pause scoring measures against, so this needs no whisper pass and no
    threshold that scoring does not already use. A rollout that never emits text
    puts all of its above-gate audio in the "before" bucket, which is the point:
    that is the audio the model produced with its text head silent.
    """
    blocks = [
        PlayedAudio(
            start_sec=audio["start_sec"] - origin,
            end_sec=audio["end_sec"] - origin,
            rms=audio["rms"],
        )
        for audio in trace["played_audio"]
    ]
    cut = float("inf") if first_text_sec is None else first_text_sec
    before = after = 0.0
    for start, end in speech_intervals(blocks):
        before += max(0.0, min(end, cut) - start)
        after += max(0.0, end - max(start, cut))
    return before, after


def gated_sec_within(trace: dict, origin: float, horizon_sec: float) -> float:
    """Seconds above the gate in the first `horizon_sec` of model audio.

    A longer drain cannot change what the model did early, so this is the window
    where conditions are directly comparable regardless of rollout length.
    """
    blocks = [
        PlayedAudio(
            start_sec=audio["start_sec"] - origin,
            end_sec=audio["end_sec"] - origin,
            rms=audio["rms"],
        )
        for audio in trace["played_audio"]
    ]
    return sum(
        max(0.0, min(end, horizon_sec) - start) for start, end in speech_intervals(blocks)
    )


def rollout_row(rollout_dir: Path) -> dict | None:
    trace = json.loads((rollout_dir / "trace.json").read_text())
    events = trace["events"]
    audio = [event for event in events if event["kind"] == "model_audio_received"]
    if not audio:
        return None
    sent = [event for event in events if event["kind"] == "input_audio_sent"]
    primed = [event for event in events if event["kind"] == "duplex_primed"]
    texts = [event for event in events if event["kind"] == "model_text"]
    origin = audio[0]["at_sec"]
    streamed_sec = sum(event.get("duration_sec") or 0.0 for event in sent)
    model_sec = sum(event.get("duration_sec") or 0.0 for event in audio)
    pcm, rate = read(rollout_dir / "response.wav")
    score = json.loads((rollout_dir / "score.json").read_text())
    first_text_sec = texts[0]["at_sec"] - origin if texts else None
    gated_before, gated_after = gated_sec_split(trace, origin, first_text_sec)
    return {
        "point_id": trace["point_id"],
        "rollout": rollout_dir.name,
        "response_sec": pcm.shape[0] / rate,
        "streamed_sec": streamed_sec,
        "model_sec": model_sec,
        "output_input_ratio": model_sec / streamed_sec if streamed_sec else 0.0,
        "max_lag_sec": max(
            (
                event["at_sec"] - event["data"]["scheduled_at_sec"]
                for event in sent
                if "data" in event
            ),
            default=0.0,
        ),
        "history_sec": primed[0]["data"]["history_duration_sec"] if primed else None,
        "truncated_frames": primed[0]["data"]["truncated_frames"] if primed else None,
        "realtime_factor": primed[0]["data"]["realtime_factor"] if primed else None,
        # Relative to the first model audio frame, so a slow connection does not
        # count against the model's warm-up.
        "observed_sec": audio[-1]["at_sec"] + (audio[-1]["duration_sec"] or 0.0) - origin,
        "first_text_sec": first_text_sec,
        "gated_before_text_sec": gated_before,
        "gated_after_text_sec": gated_after,
        "gated_early_sec": gated_sec_within(trace, origin, MATCHED_WINDOW_SEC),
        "text_tokens": len(texts),
        "text": (trace.get("text") or "").strip(),
        "speech_ratio": float(speech_ratio(pcm, rate)),
        "score": score.get("score"),
        "status": score.get("status"),
    }


def load_condition(name: str) -> list[dict]:
    rows = []
    for trace_path in sorted((RESULTS / name).glob("*/rollout_*/trace.json")):
        row = rollout_row(trace_path.parent)
        if row is not None:
            row["condition"] = name
            rows.append(row)
    return rows


def health(name: str, rows: list[dict]) -> dict:
    """Derived from the per-rollout artifacts, not from results.json.

    A DNS outage cost padtext and hist4 some rollouts, and re-running those
    points rewrote each run's results.json with only the patched points, so the
    rollout directories are the only complete record of what actually ran.
    """
    results = json.loads((RESULTS / name / "results.json").read_text())
    ratios = np.array([row["output_input_ratio"] for row in rows])
    scores = [row["score"] for row in rows if row["status"] == "scored"]
    return {
        "condition": name,
        "points": len({row["point_id"] for row in rows}),
        "rollouts": len(rows),
        "scored": len(scores),
        "unscored": sum(1 for row in rows if row["status"] != "scored"),
        "mean_score": float(np.mean(scores)) if scores else float("nan"),
        "min_output_input_ratio": float(ratios.min()) if ratios.size else 0.0,
        "max_lag_ms": max(row["max_lag_sec"] for row in rows) * 1000,
        "truncated_frames": sum(row["truncated_frames"] or 0 for row in rows),
        "median_history_sec": float(
            np.median([row["history_sec"] for row in rows if row["history_sec"] is not None])
        ),
        "conditions": results.get("conditions"),
    }


def emitted_by(rows: list[dict], horizon_sec: float) -> np.ndarray:
    """Whether each rollout had produced a text token by `horizon_sec`."""
    return np.array(
        [
            row["first_text_sec"] is not None and row["first_text_sec"] <= horizon_sec
            for row in rows
        ]
    )


def main() -> None:
    report = {name: load_condition(name) for name in CONDITIONS}
    (OUT / "experiments.json").write_text(json.dumps(report))

    print("Run health (a clean number is what a broken run used to look like)")
    print(
        f"  {'condition':14s} {'pts':>4} {'n':>3} {'scored':>6} {'unsc':>4} {'score':>6} "
        f"{'min out/in':>10} {'max lag':>8} {'trunc':>6} {'hist_s':>7}"
    )
    for name, rows in report.items():
        if not rows:
            continue
        entry = health(name, rows)
        print(
            f"  {name:14s} {entry['points']:4d} {entry['rollouts']:3d} {entry['scored']:6d} "
            f"{entry['unscored']:4d} {entry['mean_score']:6.3f} "
            f"{entry['min_output_input_ratio']:10.2f} {entry['max_lag_ms']:7.0f}ms "
            f"{entry['truncated_frames']:6d} {entry['median_history_sec']:7.2f}"
        )

    print("\nTime from the first model audio frame to the first text token")
    print(
        f"  {'condition':14s} {'n':>3} {'any text':>9} {f'by {REPORT_HORIZON_SEC:.0f}s':>8} "
        f"{'p25':>6} {'p50':>6} {'p75':>6} {'p90':>6} {'max':>6} {'obs_s':>6}"
    )
    for name, rows in report.items():
        if not rows:
            continue
        values = np.array(
            [row["first_text_sec"] for row in rows if row["first_text_sec"] is not None]
        )
        observed = np.array([row["observed_sec"] for row in rows])
        by_horizon = emitted_by(rows, REPORT_HORIZON_SEC)
        percentiles = (
            [np.percentile(values, q) for q in (25, 50, 75, 90)] + [values.max()]
            if values.size
            else [float("nan")] * 5
        )
        print(
            f"  {name:14s} {len(rows):3d} "
            f"{f'{values.size}/{len(rows)}':>9} {f'{int(by_horizon.sum())}/{len(rows)}':>8} "
            + " ".join(f"{value:6.2f}" for value in percentiles)
            + f" {np.median(observed):6.1f}"
        )

    print("\nShare of rollouts that have produced a text token by elapsed time")
    horizons = [1, 2, 3, 4, 6, 8, 10, 12, 15]
    print("  " + f"{'condition':14s}" + "".join(f"{h:>6}s" for h in horizons))
    for name, rows in report.items():
        if not rows:
            continue
        shares = []
        for horizon in horizons:
            usable = [row for row in rows if row["observed_sec"] >= horizon]
            shares.append(
                float(emitted_by(usable, horizon).mean()) if usable else float("nan")
            )
        print(
            f"  {name:14s}"
            + "".join(f"{share:7.2f}" for share in shares)
            + f"   (n={len(rows)})"
        )

    print("\nAbove-gate audio and the text stream behind it")
    print(
        f"  {'condition':14s} {'n':>3} {'resp_s':>7} {'gate%':>6} {'textless':>9} "
        f"{'gated+textless_s':>17} {'tokens/rollout':>15}"
    )
    for name, rows in report.items():
        if not rows:
            continue
        response = np.array([row["response_sec"] for row in rows])
        gate = np.array([row["speech_ratio"] for row in rows])
        textless = np.array([not row["text"] for row in rows])
        print(
            f"  {name:14s} {len(rows):3d} {response.sum():7.1f} {gate.mean():6.1%} "
            f"{f'{int(textless.sum())}/{len(rows)}':>9} "
            f"{float((gate * response)[textless].sum()):17.1f} "
            f"{np.mean([row['text_tokens'] for row in rows]):15.1f}"
        )

    print("\nAbove-gate audio produced while the text head was still silent")
    print(
        f"  {'condition':14s} {'n':>3} {'gated_s':>8} {'pre-text':>9} {'post-text':>10} "
        f"{'pre share':>10} {'pre_s/rollout':>14} {f'gate% first {MATCHED_WINDOW_SEC}s':>21}"
    )
    for name, rows in report.items():
        if not rows:
            continue
        before = sum(row["gated_before_text_sec"] for row in rows)
        after = sum(row["gated_after_text_sec"] for row in rows)
        early = np.mean([row["gated_early_sec"] for row in rows]) / MATCHED_WINDOW_SEC
        print(
            f"  {name:14s} {len(rows):3d} {before + after:8.1f} {before:9.1f} "
            f"{after:10.1f} {before / (before + after) if before + after else 0:10.2f} "
            f"{before / len(rows):14.2f} {early:20.1%}"
        )

    print("\nDoes history length predict the warm-up? (exp1_drain15)")
    rows = [row for row in report["exp1_drain15"] if row["first_text_sec"] is not None]
    history = np.array([row["history_sec"] for row in rows])
    first = np.array([row["first_text_sec"] for row in rows])
    order_h = np.argsort(np.argsort(history))
    order_f = np.argsort(np.argsort(first))
    rho = float(np.corrcoef(order_h, order_f)[0, 1]) if rows else float("nan")
    print(f"  spearman(history_sec, first_text_sec) = {rho:+.2f} over {len(rows)} rollouts")


if __name__ == "__main__":
    main()
