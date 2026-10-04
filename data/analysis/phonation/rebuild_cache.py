"""Repopulate windows.json for every cohort already recorded in cohorts.json.

windows.json is only a decode cache: each cohort row in cohorts.json already
carries the envelope and the labels derived from it, so no published number
depends on the cache surviving. It is worth rebuilding anyway, because without
it every script that calls detect.analyze re-decodes from scratch. Whisper runs
at temperature 0 with fixed thresholds here, so a rebuilt entry is the entry it
replaces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import OUT, analyze, load_cache, read  # noqa: E402
from bench.transcribe import load_whisper_model  # noqa: E402


def main() -> None:
    report = json.loads((OUT / "cohorts.json").read_text())
    cache = load_cache()
    model = load_whisper_model()
    missing = [
        row
        for rows in report.values()
        for row in rows
        if row["key"] not in cache and Path(row["path"]).is_file()
    ]
    print(f"{len(cache)} cached, {len(missing)} to decode", flush=True)
    for row in missing:
        pcm, rate = read(Path(row["path"]), row["start_sec"], row["end_sec"])
        analyze(model, row["key"], pcm, rate, cache)
    unreachable = sorted(
        {
            row["key"]
            for rows in report.values()
            for row in rows
            if not Path(row["path"]).is_file()
        }
    )
    print(f"{len(cache)} cached")
    if unreachable:
        print(f"{len(unreachable)} row(s) point at audio not in this checkout:")
        for key in unreachable:
            print(f"  {key}")


if __name__ == "__main__":
    main()
