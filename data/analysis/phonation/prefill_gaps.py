"""Does prefilled assistant audio with no word over it predict the warm-up?

exp2_padtext teacher-forces the assistant history audio against an all-PAD text
stream and comes out worse than exp1_drain15, not better: the model is primed
with voiced audio that no word accounts for, and then produces voiced audio that
no word accounts for. If that is the mechanism, the aligned-text prefill of
exp1_drain15 should be a weaker dose of the same thing, because Whisper word
timestamps do not tile an utterance and every frame they miss is a PAD frame
under audible assistant speech.

Measured per point from the artifacts the prefill is actually built from:
`rollout_assistant.wav` against `rollout_assistant_transcript.json`, on the same
80 ms frame grid the server steps and the same speech gate scoring uses. Word
coverage is taken from the word intervals rather than from the sentencepiece
stream, so this needs no model tokenizer; queued tokens can only extend
coverage, so it is an upper bound on coverage and a lower bound on PAD frames.

The rank probe (see "The forced text is in distribution" in AGENTS.md) narrows
this: the text head is not corrupted by prefill, it agrees with the PAD it is
forced to emit and ends prefill confident in it. That predicts the plain PAD
share of the forced stream should track the warm-up too, and that recency should
matter, so the tail shares are reported beside the unworded ones.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import OUT, ROOT, read  # noqa: E402
from experiments import load_condition  # noqa: E402
from bench.audio import MODEL_FRAME_SAMPLES  # noqa: E402
from bench.audio_metrics import rms_envelope  # noqa: E402
from bench.speech import SPEECH_RMS_THRESHOLD  # noqa: E402
from utils.timed_words import load_timed_words  # noqa: E402

POINTS = ROOT / "data/processed/example_1/points"
FRAME_SEC = MODEL_FRAME_SAMPLES / 24000


def point_gaps(point_dir: Path) -> dict:
    meta = json.loads((point_dir / "point.json").read_text())
    rollout = meta["rollout"]
    path = point_dir / rollout["assistant_audio"]
    pcm, rate = read(path)
    envelope = rms_envelope(pcm, rate, FRAME_SEC)
    voiced = envelope >= SPEECH_RMS_THRESHOLD
    words = load_timed_words(
        point_dir / rollout.get("assistant_transcript", "rollout_assistant_transcript.json")
    )
    worded = np.zeros(envelope.shape[0], dtype=bool)
    starts = np.arange(envelope.shape[0]) * FRAME_SEC
    for word in words:
        end_sec = word.end_sec if word.end_sec is not None else word.start_sec + FRAME_SEC
        worded |= (starts + FRAME_SEC > word.start_sec) & (starts < end_sec)
    unworded = voiced & ~worded
    tail_frames = int(round(2.0 / FRAME_SEC))
    return {
        "point_id": str(meta.get("id", point_dir.name)),
        # Points from one conversation share history audio byte for byte, and
        # prefill is all this measurement depends on, so the distinct histories
        # are the independent units and the point count overstates them.
        "history_key": hashlib.md5(path.read_bytes()).hexdigest()[:8],
        "history_sec": pcm.shape[0] / rate,
        "voiced_sec": float(voiced.sum()) * FRAME_SEC,
        "unworded_sec": float(unworded.sum()) * FRAME_SEC,
        "unworded_share_of_voiced": float(unworded.sum() / voiced.sum()) if voiced.any() else 0.0,
        "pad_share": float(np.mean(~worded)),
        "pad_share_last_2s": float(np.mean(~worded[-tail_frames:])),
        "words": len(words),
    }


def ranked(values: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(values)).astype(float)


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 3:
        return float("nan")
    return float(np.corrcoef(ranked(left), ranked(right))[0, 1])


def partial_spearman(
    left: np.ndarray, right: np.ndarray, control: np.ndarray
) -> float:
    """Spearman between left and right with control regressed out of both.

    The dose and the history length are collinear here, so a raw correlation
    cannot say which of the two the warm-up follows.
    """
    control_ranks = ranked(control)
    residuals = []
    for values in (ranked(left), ranked(right)):
        fit = np.polyfit(control_ranks, values, 1)
        residuals.append(values - np.polyval(fit, control_ranks))
    return float(np.corrcoef(*residuals)[0, 1])


def permutation_p(left: np.ndarray, right: np.ndarray) -> float:
    """Exact two-sided p for a spearman on a handful of units.

    Seven histories is few enough to enumerate every relabelling, which beats
    quoting a correlation with no idea how often chance produces it.
    """
    observed = abs(spearman(left, right))
    orderings = list(itertools.permutations(range(left.size)))
    hits = sum(
        abs(spearman(left, right[list(order)])) >= observed - 1e-12 for order in orderings
    )
    return hits / len(orderings)


def main() -> None:
    rows = load_condition("exp1_drain15")
    by_point: dict[str, list[float]] = {}
    for row in rows:
        if row["first_text_sec"] is not None:
            by_point.setdefault(row["point_id"], []).append(row["first_text_sec"])

    gaps = []
    for point_dir in sorted(POINTS.iterdir()):
        if not (point_dir / "point.json").is_file():
            continue
        entry = point_gaps(point_dir)
        if entry["point_id"] not in by_point:
            continue
        entry["median_first_text_sec"] = float(np.median(by_point[entry["point_id"]]))
        gaps.append(entry)
    (OUT / "prefill_gaps.json").write_text(json.dumps(gaps, indent=1))

    print("Prefilled assistant history: audible speech with no word over it")
    print(
        f"  {'point':18s} {'hist_s':>7} {'voiced_s':>9} {'unworded_s':>11} "
        f"{'unworded share':>15} {'median first_text':>18}"
    )
    for entry in sorted(gaps, key=lambda row: row["history_sec"]):
        print(
            f"  {entry['point_id']:18s} {entry['history_sec']:7.2f} "
            f"{entry['voiced_sec']:9.2f} {entry['unworded_sec']:11.2f} "
            f"{entry['unworded_share_of_voiced']:15.2f} "
            f"{entry['median_first_text_sec']:18.2f}"
        )

    distinct = []
    for key in dict.fromkeys(entry["history_key"] for entry in gaps):
        shared = [entry for entry in gaps if entry["history_key"] == key]
        distinct.append(
            shared[0]
            | {
                "median_first_text_sec": float(
                    np.median([entry["median_first_text_sec"] for entry in shared])
                )
            }
        )

    for cohort, name in ((gaps, "points"), (distinct, "distinct histories")):
        first = np.array([entry["median_first_text_sec"] for entry in cohort])
        print(
            f"\n  over {len(cohort)} {name}, spearman against median time-to-first-text:"
        )
        for key in (
            "history_sec",
            "voiced_sec",
            "unworded_sec",
            "unworded_share_of_voiced",
            "pad_share",
            "pad_share_last_2s",
        ):
            values = np.array([entry[key] for entry in cohort])
            print(f"    {key:26s} {spearman(values, first):+.2f}")
        dose = np.array([entry["unworded_share_of_voiced"] for entry in cohort])
        length = np.array([entry["history_sec"] for entry in cohort])
        print(
            f"    the two are collinear (spearman {spearman(dose, length):+.2f}), so partialled:\n"
            f"      unworded share, controlling for history length  "
            f"{partial_spearman(dose, first, length):+.2f}\n"
            f"      history length, controlling for unworded share  "
            f"{partial_spearman(length, first, dose):+.2f}"
        )
        if len(cohort) <= 8:
            print(
                f"      exact permutation p for the unworded share  "
                f"{permutation_p(dose, first):.3f}"
            )
    print(
        "\n  Total unworded voiced history across these points: "
        f"{sum(entry['unworded_sec'] for entry in gaps):.1f} s of "
        f"{sum(entry['voiced_sec'] for entry in gaps):.1f} s voiced"
    )


if __name__ == "__main__":
    main()
