"""Do the prepared assistant transcripts cover the voiced assistant history?

The prefill dose measured in `prefill_gaps.py` is voiced assistant history frames
that no stored word interval covers, and that quantity predicts the warm-up. It
has two very different possible compositions, and the fix differs by which:

- speech the prepare-time transcript missed (backchannels, partial words, quiet
  speech), which is a labeling gap and teaches the model a pairing it would not
  have seen in training;
- gaps between words inside an utterance, which every transcript has and which
  training saw too.

Separated here with the independent labeler in `detect.py`: 2 s windows at a 1 s
hop, each peak normalized, a cell counted as speech only when every window
covering it puts a word over it. That labeler is stricter about hallucination and
more generous about level than a single pass over a whole file, so a voiced cell
it calls speech while the stored transcript has no word there is under-coverage.

Reads the same `rollout_assistant.wav` and `rollout_assistant_transcript.json`
the duplex prefill is built from, and caches decodes in `windows.json`.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import CELL_SEC, OUT, ROOT, analyze, load_cache, read  # noqa: E402
from bench.transcribe import load_whisper_model  # noqa: E402
from utils.timed_words import load_timed_words  # noqa: E402

POINTS = ROOT / "data/processed/example_1/points"
# Below this a voiced unlabeled run is an inter-word gap; a plosive closure or a
# breath between words is tens of milliseconds, not half a second.
LONG_RUN_SEC = 0.5


def transcript_cover(point_dir: Path, rollout: dict, cells: int) -> np.ndarray:
    words = load_timed_words(
        point_dir / rollout.get("assistant_transcript", "rollout_assistant_transcript.json")
    )
    centers = np.arange(cells) * CELL_SEC + CELL_SEC / 2
    covered = np.zeros(cells, dtype=bool)
    for word in words:
        end_sec = word.end_sec if word.end_sec is not None else word.start_sec + CELL_SEC
        covered |= (centers >= word.start_sec) & (centers < end_sec)
    return covered


def runs_of(flags: np.ndarray) -> list[tuple[int, int]]:
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2], edges[1::2], strict=True))


def point_coverage(model, cache: dict, point_dir: Path) -> dict:
    meta = json.loads((point_dir / "point.json").read_text())
    rollout = meta["rollout"]
    path = point_dir / rollout["assistant_audio"]
    pcm, rate = read(path)
    # Points from one conversation share history audio byte for byte, so keying
    # the decode on content rather than point id keeps whisper off the duplicates.
    key = f"assistant_history/{hashlib.md5(path.read_bytes()).hexdigest()[:12]}"
    analysis = analyze(model, key, pcm, rate, cache)
    gated = analysis.gated()
    covered = transcript_cover(point_dir, rollout, analysis.cells)
    missed = gated & ~covered & analysis.labeled
    gap = gated & ~covered & ~analysis.labeled
    long_missed = np.zeros(analysis.cells, dtype=bool)
    for start, end in runs_of(gated & ~covered):
        if (end - start) * CELL_SEC >= LONG_RUN_SEC:
            long_missed[start:end] = True
    return {
        "point_id": str(meta.get("id", point_dir.name)),
        "history_sec": pcm.shape[0] / rate,
        "gated_sec": float(gated.sum()) * CELL_SEC,
        "covered_sec": float((gated & covered).sum()) * CELL_SEC,
        "unworded_sec": float((gated & ~covered).sum()) * CELL_SEC,
        "missed_speech_sec": float(missed.sum()) * CELL_SEC,
        "unlabeled_gap_sec": float(gap.sum()) * CELL_SEC,
        "long_run_sec": float(long_missed.sum()) * CELL_SEC,
        "unworded_runs": [
            round((end - start) * CELL_SEC, 3) for start, end in runs_of(gated & ~covered)
        ],
    }


def main() -> None:
    cache = load_cache()
    model = load_whisper_model()
    rows = []
    for point_dir in sorted(POINTS.iterdir()):
        if not (point_dir / "point.json").is_file():
            continue
        rows.append(point_coverage(model, cache, point_dir))
        print(f"  {rows[-1]['point_id']} done", flush=True)
    (OUT / "transcript_coverage.json").write_text(json.dumps(rows, indent=1))

    print("\nVoiced assistant history the stored transcript does not cover")
    print(
        f"  {'point':18s} {'hist_s':>7} {'voiced_s':>9} {'unworded_s':>11} "
        f"{'missed speech':>14} {'unlabeled gap':>14} {'in runs >= 0.5 s':>17}"
    )
    for row in rows:
        print(
            f"  {row['point_id']:18s} {row['history_sec']:7.2f} {row['gated_sec']:9.2f} "
            f"{row['unworded_sec']:11.2f} {row['missed_speech_sec']:14.2f} "
            f"{row['unlabeled_gap_sec']:14.2f} {row['long_run_sec']:17.2f}"
        )

    totals = {
        key: sum(row[key] for row in rows)
        for key in ("gated_sec", "unworded_sec", "missed_speech_sec", "unlabeled_gap_sec", "long_run_sec")
    }
    every_run = np.array([run for row in rows for run in row["unworded_runs"]])
    print(
        f"\n  {totals['unworded_sec']:.1f} s of {totals['gated_sec']:.1f} s voiced "
        f"assistant history carries no stored word "
        f"({totals['unworded_sec'] / totals['gated_sec']:.1%})"
    )
    print(
        f"  of that, {totals['missed_speech_sec']:.1f} s "
        f"({totals['missed_speech_sec'] / totals['unworded_sec']:.1%}) is speech the "
        f"independent labeler does find, and {totals['unlabeled_gap_sec']:.1f} s is not"
    )
    print(
        f"  {totals['long_run_sec']:.1f} s "
        f"({totals['long_run_sec'] / totals['unworded_sec']:.1%}) sits in runs of "
        f"{LONG_RUN_SEC:g} s or longer, too long to be an inter-word gap"
    )
    print(
        f"  {every_run.shape[0]} runs: median {np.median(every_run) * 1000:.0f} ms, "
        f"p90 {np.percentile(every_run, 90) * 1000:.0f} ms, "
        f"max {every_run.max() * 1000:.0f} ms"
    )


if __name__ == "__main__":
    main()
