"""Pull a mirrored eval run from S3 into data/results/ and rebuild summaries.

bench/eval_modal.py mirrors every rollout to
s3://pmt-data-seamless/results/<run_id>/ as it finishes. This downloads the
run tree (skipping files already present with the same size, so it is safe to
run mid-run and again after) and rebuilds run-level results.json and figures
from the synced scores.

Run: uv run python scripts/sync_results.py <run_id>
"""

from __future__ import annotations

import sys
from pathlib import Path

from bench.report.summarize import write_run_summaries
from bench.results_s3 import ResultsMirror

RESULTS_ROOT = Path("data/results")


def main() -> None:
    run_id = sys.argv[1]
    run_dir = RESULTS_ROOT / run_id
    downloaded = ResultsMirror(run_dir).download(skip_same_size=True)
    print(f"{downloaded} file(s) synced")

    _results, paths = write_run_summaries(run_dir, manifest=None, point=None)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
