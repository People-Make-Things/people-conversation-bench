"""Rescore stored recall transcripts with a different judge.

Reads conversation-recall and fact-recall score.json files from a run and
writes a parallel run directory. Source scores stay in place. Unscored
rollouts and error.json files are copied unchanged, so a later
`bench report` on the new directory counts the same rollouts.

Run: uv run python scripts/rescore_recall.py data/results/<run_id>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bench.metrics.recall import rescore_recall_tree
from utils.anthropic import anthropic_json_complete

DEFAULT_MODEL = "claude-opus-5-5"
RESULTS_ROOT = Path("data/results")


def resolve_run_dir(value: str) -> Path:
    path = Path(value)
    if path.is_dir():
        return path
    under_results = RESULTS_ROOT / value
    if under_results.is_dir():
        return under_results
    raise SystemExit(f"run directory not found: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Run directory, or an id under data/results")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--output",
        type=Path,
        help="Parallel run directory (default: <run>-<model>)",
    )
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    run_dir = resolve_run_dir(args.run)
    output_dir = args.output or run_dir.parent / f"{run_dir.name}-{args.model}"
    counts = rescore_recall_tree(
        run_dir,
        output_dir,
        model=args.model,
        complete=anthropic_json_complete,
        workers=args.workers,
    )
    print(
        f"Done. judged {counts['judged']}, copied {counts['copied']}, "
        f"skipped {counts['skipped']} -> {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
