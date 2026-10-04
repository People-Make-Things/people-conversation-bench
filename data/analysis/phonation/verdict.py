"""Does the model itself think it said words, and how much audio has no text at all?

PersonaPlex emits a text stream alongside audio, and the harness records it in
trace.json as `model_text` events plus the joined `text`. That is a label source
completely independent of Whisper and of any RMS threshold: it is the model's own
account of what it was saying. Cross-checking the two answers "is Whisper failing
to transcribe backchannels" without having to trust Whisper.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import CELL_SEC, OUT, ROOT, analyze, load_cache, read  # noqa: E402
from bench.transcribe import load_whisper_model  # noqa: E402

MAIN = ROOT.parent / "people-bench" / "data" / "results"
RUN_DIRS = {
    "model_pause": ROOT / "data/results/personaplex/baseline_v2",
    "model_eot": ROOT / "data/results/personaplex/eot_baseline_10",
    "model_pause_prefix": ROOT / "data/results/personaplex/baseline",
    "model_pause_voiceprompt": MAIN / "personaplex/20260820T001008Z",
}


def rollout_rows(cohort: str, rows: list[dict], model, cache: dict) -> list[dict]:
    run = RUN_DIRS[cohort]
    out = []
    for row in rows:
        point, rollout = row["key"].split("/")[-2:]
        trace = json.loads((run / point / rollout / "trace.json").read_text())
        pcm, rate = read(Path(row["path"]))
        analysis = analyze(model, row["key"], pcm, rate, cache)
        text = (trace.get("text") or "").strip()
        tokens = sum(1 for e in trace["events"] if e["kind"] == "model_text")
        out.append(
            {
                "cohort": cohort,
                "point": point,
                "rollout": rollout,
                "duration_sec": row["duration_sec"],
                "gated_sec": row["gated_sec"],
                "lexical_sec": row["labeled_sec"],
                "phonation_sec": row["phonation_sec"],
                "gated_fraction": row["gated_sec"] / row["duration_sec"],
                "model_text": text,
                "model_text_tokens": tokens,
                "whisper_words": int(
                    sum(
                        1
                        for window in analysis.windows
                        for word in window["words"]
                        if word["end"] > word["start"]
                    )
                    > 0
                ),
                "any_lexical": bool(analysis.labeled.any()),
            }
        )
    return out


def confusion(rows: list[dict], label: str) -> None:
    both = sum(1 for r in rows if r["model_text"] and r["any_lexical"])
    text_only = sum(1 for r in rows if r["model_text"] and not r["any_lexical"])
    whisper_only = sum(1 for r in rows if not r["model_text"] and r["any_lexical"])
    neither = sum(1 for r in rows if not r["model_text"] and not r["any_lexical"])
    loud_neither = sum(
        1
        for r in rows
        if not r["model_text"] and not r["any_lexical"] and r["gated_sec"] > 0
    )
    print(f"\n{label} (n={len(rows)})")
    print("                        whisper words   no whisper words")
    print(f"  model emitted text   {both:13d} {text_only:18d}")
    print(f"  model emitted none   {whisper_only:13d} {neither:18d}")
    print(
        f"  -> {neither} rollouts have neither, of which {loud_neither} still put audio "
        f"above the 0.01 gate"
    )
    silent = [r for r in rows if not r["model_text"]]
    if silent:
        gated = np.array([r["gated_sec"] for r in silent])
        fraction = np.array([r["gated_fraction"] for r in silent])
        print(
            f"  in the {len(silent)} no-text rollouts: above-gate audio "
            f"{gated.sum():.1f}s total, {np.median(fraction):.1%} of the rollout (median)"
        )


def main() -> None:
    report = json.loads((OUT / "cohorts.json").read_text())
    cache = load_cache()
    model = load_whisper_model()
    rows: dict[str, list[dict]] = {
        cohort: rollout_rows(cohort, report[cohort], model, cache)
        for cohort in RUN_DIRS
    }
    (OUT / "verdict.json").write_text(json.dumps(rows))

    for cohort, label in (
        ("model_pause", "PersonaPlex pause rollouts (baseline_v2)"),
        ("model_eot", "PersonaPlex EOT rollouts (eot_baseline_10)"),
        ("model_pause_prefix", "PersonaPlex pause rollouts, pre-fix (baseline)"),
        ("model_pause_voiceprompt", "PersonaPlex pause, voice_prompt='NATF2.pt'"),
    ):
        confusion(rows[cohort], label)

    print("\n\nVoice-prompt A/B, restricted to the points both runs cover")
    cleared = {(r["point"], r["rollout"]): r for r in rows["model_pause"]}
    prompted = {(r["point"], r["rollout"]): r for r in rows["model_pause_voiceprompt"]}
    shared = sorted(set(cleared) & set(prompted))
    print(f"  {len(shared)} matched rollouts")
    print(
        f"  {'variant':28s} {'dur_s':>8} {'gated_s':>8} {'gated%':>7} {'lex_s':>7} "
        f"{'phon_s':>8} {'text rollouts':>14}"
    )
    for name, table in (("voice_prompt=NATF2.pt", prompted), ("voice_prompt='' (current)", cleared)):
        picked = [table[key] for key in shared]
        duration = sum(r["duration_sec"] for r in picked)
        gated = sum(r["gated_sec"] for r in picked)
        spoke = sum(1 for r in picked if r["model_text"])
        print(
            f"  {name:28s} {duration:8.1f} {gated:8.1f} {gated / duration:7.1%} "
            f"{sum(r['lexical_sec'] for r in picked):7.1f} "
            f"{sum(r['phonation_sec'] for r in picked):8.1f} "
            f"{f'{spoke}/{len(picked)}':>14}"
        )

    print("\n\nPersonaPlex pause: above-gate audio split by whether the model was")
    print("emitting text at all")
    for cohort in ("model_pause", "model_eot"):
        with_text = [r for r in rows[cohort] if r["model_text"]]
        without = [r for r in rows[cohort] if not r["model_text"]]
        for label, group in (("has text", with_text), ("no text", without)):
            if not group:
                continue
            duration = sum(r["duration_sec"] for r in group)
            gated = sum(r["gated_sec"] for r in group)
            print(
                f"  {cohort:14s} {label:9s} n={len(group):3d} dur={duration:7.1f}s "
                f"gated={gated:6.1f}s ({gated / duration:5.1%}) "
                f"lex={sum(r['lexical_sec'] for r in group):5.1f}s "
                f"phon={sum(r['phonation_sec'] for r in group):6.1f}s"
            )


if __name__ == "__main__":
    main()
