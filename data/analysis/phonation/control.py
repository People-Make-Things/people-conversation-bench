"""Human-audio control for the "wordless above-gate phonation" finding.

Runs the identical detector over four cohorts:

  model_pause   baseline_v2 response.wav        (19 pause points x 3 rollouts)
  model_eot     eot_baseline_10 response.wav
  human_live    assistant.wav over exactly the live window each pause rollout
                is scored on: [history_end, input_end] = turn start -> pause_end
  human_eot     rollout_reference_response.wav, the human turn the EOT metric
                compares against
  human_full    each speaker's whole channel, as a broad human baseline

The human_live cohort is co-timed and duration-matched with model_pause point by
point, which is what makes the comparison mean anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import (  # noqa: E402
    CACHE,
    CELL_SEC,
    OUT,
    ROOT,
    analyze,
    load_cache,
    read,
)
from bench.transcribe import load_whisper_model  # noqa: E402

RESULTS = ROOT / "data/results/personaplex"
POINTS = ROOT / "data/processed/example_1/points"
EOT_POINTS = ROOT / "data/processed_eot/example_1/points"
LEGACY_CACHE = Path("/tmp/cal/windows.json")


def seed_cache(cache: dict) -> dict:
    if not LEGACY_CACHE.is_file():
        return cache
    for key, value in json.loads(LEGACY_CACHE.read_text()).items():
        cache.setdefault(key, value)
    return cache


def point_meta(point_dir: Path) -> dict:
    return json.loads((point_dir / "point.json").read_text())


def cohorts() -> dict[str, list[tuple[str, Path, float, float | None]]]:
    """Cohort name -> (key, wav path, start_sec, end_sec)."""
    groups: dict[str, list[tuple[str, Path, float, float | None]]] = {}

    for run, name in (
        ("baseline_v2", "model_pause"),
        ("eot_baseline_10", "model_eot"),
        ("baseline", "model_pause_prefix"),
    ):
        groups[name] = [
            (
                f"{run}/{path.parent.parent.name}/{path.parent.name}",
                path,
                0.0,
                None,
            )
            for path in sorted((RESULTS / run).glob("*/rollout_*/response.wav"))
        ]

    live: list[tuple[str, Path, float, float | None]] = []
    for point_dir in sorted(POINTS.iterdir()):
        meta = point_meta(point_dir)
        rollout = meta["rollout"]
        if rollout["expected_action"] != "pause":
            continue
        live.append(
            (
                f"human_live/{meta['id']}",
                point_dir / "assistant.wav",
                rollout["history_end_sec"],
                rollout["input_end_sec"],
            )
        )
    groups["human_live"] = live

    groups["human_eot"] = [
        (f"human_eot/{point_dir.name}", point_dir / "rollout_reference_response.wav", 0.0, None)
        for point_dir in sorted(EOT_POINTS.iterdir())
        if (point_dir / "rollout_reference_response.wav").is_file()
    ]

    deepest = sorted(POINTS.iterdir())[-1]
    groups["human_full"] = [
        ("human_full/speaker_assistant", deepest / "assistant.wav", 0.0, None),
        ("human_full/speaker_user", deepest / "user.wav", 0.0, None),
    ]
    return groups


def main() -> None:
    cache = seed_cache(load_cache())
    CACHE.write_text(json.dumps(cache))
    model = load_whisper_model()
    report: dict[str, list[dict]] = {}
    for name, items in cohorts().items():
        rows = []
        for key, path, start, end in items:
            pcm, rate = read(path, start, end)
            if pcm.shape[0] < int(0.1 * rate):
                continue
            analysis = analyze(model, key, pcm, rate, cache)
            row = analysis.summary()
            row["cohort"] = name
            row["path"] = str(path)
            row["start_sec"] = start
            row["end_sec"] = end if end is not None else analysis.duration_sec
            row["envelope"] = [round(float(v), 6) for v in analysis.envelope]
            row["labeled"] = [bool(v) for v in analysis.labeled]
            rows.append(row)
        report[name] = rows
        print(f"{name}: {len(rows)} segments", flush=True)
    (OUT / "cohorts.json").write_text(json.dumps(report))

    print()
    header = (
        f"{'cohort':22s} {'n':>3} {'dur_s':>8} {'gated_s':>8} {'lex_s':>8} "
        f"{'phon_s':>8} {'phon/gated':>10} {'files wordless':>15}"
    )
    print(header)
    for name, rows in report.items():
        if not rows:
            continue
        duration = sum(r["duration_sec"] for r in rows)
        gated = sum(r["gated_sec"] for r in rows)
        lexical = sum(r["labeled_sec"] for r in rows)
        phonation = sum(r["phonation_sec"] for r in rows)
        wordless = sum(1 for r in rows if r["gated_but_wordless_file"])
        print(
            f"{name:22s} {len(rows):3d} {duration:8.1f} {gated:8.1f} {lexical:8.1f} "
            f"{phonation:8.1f} {phonation / gated if gated else 0:10.3f} "
            f"{f'{wordless}/{len(rows)}':>15}"
        )


if __name__ == "__main__":
    main()
