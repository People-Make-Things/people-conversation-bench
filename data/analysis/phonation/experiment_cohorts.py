"""Run the four experiment conditions through the shared phonation detector.

Same code and same cache as the cohorts already in cohorts.json, so
`phonation_fraction_of_gated` here is directly comparable to the 0.909 that
baseline_v2 scores and the 0.272 that GPT Realtime scores. Appends new cohorts
rather than rewriting the existing ones, so the old numbers stay reproducible.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

import numpy as np  # noqa: E402

from detect import CELL_SEC, OUT, analyze, load_cache, read  # noqa: E402
from experiments import CONDITIONS, RESULTS, rollout_row  # noqa: E402
from bench.transcribe import load_whisper_model  # noqa: E402

COHORTS = [name for name in CONDITIONS if name != "baseline_v2"]


def main() -> None:
    cache = load_cache()
    model = load_whisper_model()
    report = json.loads((OUT / "cohorts.json").read_text())

    for name in COHORTS:
        rows = []
        for path in sorted((RESULTS / name).glob("*/rollout_*/response.wav")):
            key = f"{name}/{path.parent.parent.name}/{path.parent.name}"
            pcm, rate = read(path)
            if pcm.shape[0] < int(0.1 * rate):
                continue
            analysis = analyze(model, key, pcm, rate, cache)
            row = analysis.summary()
            row["cohort"] = name
            row["path"] = str(path)
            row["start_sec"] = 0.0
            row["end_sec"] = analysis.duration_sec
            row["model_text"] = (
                json.loads((path.parent / "trace.json").read_text()).get("text") or ""
            ).strip()
            rows.append(row)
        report[name] = rows
        print(f"{name}: {len(rows)} rollouts", flush=True)
    (OUT / "cohorts.json").write_text(json.dumps(report))

    print()
    print(
        f"{'cohort':16s} {'n':>3} {'dur_s':>8} {'gated_s':>8} {'lex_s':>8} {'phon_s':>8} "
        f"{'phon/gated':>10} {'gated%':>7} {'wordless files':>14} {'no text':>8}"
    )
    for name in ("model_pause", "model_eot", "model_pause_gpt", *COHORTS):
        rows = report.get(name) or []
        if not rows:
            continue
        duration = sum(row["duration_sec"] for row in rows)
        gated = sum(row["gated_sec"] for row in rows)
        lexical = sum(row["labeled_sec"] for row in rows)
        phonation = sum(row["phonation_sec"] for row in rows)
        wordless = sum(1 for row in rows if row["gated_but_wordless_file"])
        texts = [row["model_text"] for row in rows if "model_text" in row]
        print(
            f"{name:16s} {len(rows):3d} {duration:8.1f} {gated:8.1f} {lexical:8.1f} "
            f"{phonation:8.1f} {phonation / gated if gated else 0:10.3f} "
            f"{gated / duration if duration else 0:7.1%} "
            f"{f'{wordless}/{len(rows)}':>14} "
            f"{f'{sum(1 for t in texts if not t)}/{len(texts)}' if texts else '-':>8}"
        )

    split_table(cache, model)


def split_table(cache: dict, model) -> None:
    """Audio before and after the first text token, in rollouts that emitted one.

    If the silent text head is a warm-up the model comes out of, the audio after
    the first token should look like the EOT cohort and the audio before it
    should carry the wordless phonation.
    """
    print("\nWithin a rollout that did emit text: before vs after the first token")
    print(
        f"{'cohort':16s} {'n':>3} {'pre_s':>7} {'pre gate%':>10} {'pre phon/gated':>15} "
        f"{'post_s':>7} {'post gate%':>11} {'post phon/gated':>16}"
    )
    for name in COHORTS:
        pre = {"dur": 0.0, "gated": 0.0, "phon": 0.0}
        post = {"dur": 0.0, "gated": 0.0, "phon": 0.0}
        count = 0
        for path in sorted((RESULTS / name).glob("*/rollout_*/response.wav")):
            row = rollout_row(path.parent)
            if row is None or row["first_text_sec"] is None:
                continue
            pcm, rate = read(path)
            analysis = analyze(
                model, f"{name}/{path.parent.parent.name}/{path.parent.name}", pcm, rate, cache
            )
            gated = analysis.gated()
            phonation = analysis.phonation()
            cut = min(int(round(row["first_text_sec"] / CELL_SEC)), analysis.cells)
            count += 1
            for bucket, sl in ((pre, slice(0, cut)), (post, slice(cut, None))):
                bucket["dur"] += float(gated[sl].shape[0]) * CELL_SEC
                bucket["gated"] += float(gated[sl].sum()) * CELL_SEC
                bucket["phon"] += float(phonation[sl].sum()) * CELL_SEC
        if not count:
            continue
        print(
            f"{name:16s} {count:3d} {pre['dur']:7.1f} "
            f"{pre['gated'] / pre['dur'] if pre['dur'] else 0:10.1%} "
            f"{pre['phon'] / pre['gated'] if pre['gated'] else 0:15.3f} "
            f"{post['dur']:7.1f} "
            f"{post['gated'] / post['dur'] if post['dur'] else 0:11.1%} "
            f"{post['phon'] / post['gated'] if post['gated'] else 0:16.3f}"
        )


if __name__ == "__main__":
    main()
