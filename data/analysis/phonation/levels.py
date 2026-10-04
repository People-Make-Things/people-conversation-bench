"""Level anchoring and cross-channel listening behaviour.

Two things a naive cohort comparison gets wrong:

1. The absolute 0.01 gate is only comparable across cohorts if their speech sits
   at the same level. Anchor on the RMS of cells the labeler calls lexical
   speech, then re-run the comparison with each cohort scaled to a common
   anchor.
2. "Human backchannels" only show up while the other speaker holds the floor.
   Both human channels of the same conversation are on disk, so listening time
   can be defined by the other channel's lexical labels rather than guessed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import CELL_SEC, OUT, ROOT, analyze, load_cache, read, runs  # noqa: E402
from bench.speech import SPEECH_RMS_THRESHOLD  # noqa: E402
from bench.transcribe import load_whisper_model  # noqa: E402

POINTS = ROOT / "data/processed/example_1/points"


def percentiles(values: np.ndarray) -> str:
    if values.size == 0:
        return "         (none)"
    return " ".join(
        f"{np.percentile(values, p):8.2e}" for p in (10, 25, 50, 75, 90)
    )


def main() -> None:
    report = json.loads((OUT / "cohorts.json").read_text())

    print("RMS of 20 ms cells, by cohort and label (p10 p25 p50 p75 p90)")
    print(f"{'cohort':22s} {'cells':>6} {'kind':10s} {'p10':>8} {'p25':>8} {'p50':>8} {'p75':>8} {'p90':>8}")
    anchors: dict[str, float] = {}
    for name, rows in report.items():
        envelope = np.concatenate([np.array(r["envelope"]) for r in rows])
        labeled = np.concatenate([np.array(r["labeled"], dtype=bool) for r in rows])
        lexical = envelope[labeled]
        anchors[name] = float(np.median(lexical)) if lexical.size else float("nan")
        for kind, values in (("lexical", lexical), ("unlabeled", envelope[~labeled])):
            print(
                f"{name:22s} {values.size:6d} {kind:10s} {percentiles(values)}"
            )

    print("\nlexical-cell median RMS (the level anchor):")
    for name, value in anchors.items():
        print(f"  {name:22s} {value:.4e}")

    target = anchors["human_full"]
    print(f"\nRe-running the cohort comparison with every cohort scaled so its")
    print(f"lexical-cell median RMS equals human_full's ({target:.4e}):")
    print(
        f"{'cohort':22s} {'scale':>7} {'dur_s':>8} {'gated_s':>8} {'lex_s':>8} "
        f"{'phon_s':>8} {'phon/gated':>10} {'wordless files':>15}"
    )
    scaled_summary: dict[str, dict] = {}
    for name, rows in report.items():
        scale = target / anchors[name] if anchors[name] > 0 else 1.0
        duration = gated_sec = lexical_sec = phonation_sec = 0.0
        wordless = 0
        for row in rows:
            envelope = np.array(row["envelope"]) * scale
            labeled = np.array(row["labeled"], dtype=bool)
            gated = gate_cells(envelope)
            phonation = gated & ~labeled
            duration += row["duration_sec"]
            gated_sec += float(gated.sum()) * CELL_SEC
            lexical_sec += float(labeled.sum()) * CELL_SEC
            phonation_sec += float(phonation.sum()) * CELL_SEC
            wordless += int(gated.any() and not labeled.any())
        scaled_summary[name] = {
            "scale": scale,
            "duration_sec": duration,
            "gated_sec": gated_sec,
            "lexical_sec": lexical_sec,
            "phonation_sec": phonation_sec,
            "phonation_over_gated": phonation_sec / gated_sec if gated_sec else 0.0,
            "wordless_files": wordless,
            "files": len(rows),
        }
        print(
            f"{name:22s} {scale:7.2f} {duration:8.1f} {gated_sec:8.1f} "
            f"{lexical_sec:8.1f} {phonation_sec:8.1f} "
            f"{phonation_sec / gated_sec if gated_sec else 0:10.3f} "
            f"{f'{wordless}/{len(rows)}':>15}"
        )

    listening = cross_channel()
    (OUT / "levels.json").write_text(
        json.dumps(
            {
                "anchors": anchors,
                "scaled": scaled_summary,
                "listening": listening,
            }
        )
    )


def gate_cells(envelope: np.ndarray, threshold: float = SPEECH_RMS_THRESHOLD) -> np.ndarray:
    """Hysteresis gate over a cell envelope; same rule as bench.speech."""
    release = threshold * 0.5
    flags = np.zeros(envelope.shape[0], dtype=bool)
    open_at: int | None = None
    for index, value in enumerate(envelope):
        limit = release if open_at is not None else threshold
        if value >= limit:
            if open_at is None:
                open_at = index
            flags[index] = True
            continue
        if open_at is not None and index - open_at < 4:
            flags[open_at:index] = False
        open_at = None
    if open_at is not None and envelope.shape[0] - open_at < 4:
        flags[open_at:] = False
    return flags


def cross_channel() -> dict:
    """What does each human speaker's channel do while the other holds the floor?"""
    cache = load_cache()
    model = load_whisper_model()
    deepest = sorted(POINTS.iterdir())[-1]
    channels = {}
    for label, name in (("speaker_user", "user.wav"), ("speaker_assistant", "assistant.wav")):
        pcm, rate = read(deepest / name)
        channels[label] = analyze(model, f"human_full/{label}", pcm, rate, cache)

    print("\nHuman listening behaviour (each channel while the other is speaking):")
    out = {}
    names = list(channels)
    for label in names:
        other = channels[names[1 - names.index(label)]]
        me = channels[label]
        length = min(me.cells, other.cells)
        listening = other.labeled[:length] & ~me.labeled[:length]
        gated = me.gated()[:length]
        phonation = gated & listening
        spans = runs(phonation)
        durations = [(end - start) * CELL_SEC for start, end in spans]
        out[label] = {
            "listening_sec": float(listening.sum()) * CELL_SEC,
            "gated_while_listening_sec": float(phonation.sum()) * CELL_SEC,
            "events": len(spans),
            "event_durations_sec": durations,
            "events_per_min_listening": (
                len(spans) / (float(listening.sum()) * CELL_SEC / 60)
                if listening.any()
                else 0.0
            ),
        }
        print(
            f"  {label:20s} listening={out[label]['listening_sec']:6.2f}s "
            f"above-gate-while-listening={out[label]['gated_while_listening_sec']:5.2f}s "
            f"events={len(spans)} "
            f"({out[label]['events_per_min_listening']:.1f}/min) "
            f"durations={[round(d, 2) for d in durations]}"
        )
    return out


if __name__ == "__main__":
    main()
