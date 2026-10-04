"""CLI for preparing eval datapoints from annotated conversations."""

from __future__ import annotations

import argparse
import sys

from bench.transcribe import require_whisper
from data_processing.curation import DEFAULT_KEEP_PER_TYPE, run_curation
from data_processing.curation_judge import DEFAULT_JUDGE_MODEL, judge_curation
from data_processing.prepare_eval import run
from data_processing.sources import DEFAULT_DATASOURCE
from data_processing.tts import require_tts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build eval context datapoints from an S3 data source",
    )
    parser.add_argument(
        "datapoint",
        nargs="*",
        help="Specific datapoint(s) to process, for example: example_1",
    )
    parser.add_argument(
        "--datasource",
        default=DEFAULT_DATASOURCE,
        help=(
            "Data source to load (default: annotated). "
            "Use seamless for pmt-data-seamless."
        ),
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output directory for processed eval points (default: data/processed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N datapoints from the source",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="Write at most N pause/EOT points per datapoint (skips recall)",
    )
    parser.add_argument(
        "--max-datapoints",
        type=int,
        default=None,
        help="Stop after N session chunks (a datapoint is one clean-cut chunk)",
    )
    parser.add_argument(
        "--profile",
        default="pmt",
        help="AWS profile (default: pmt)",
    )
    parser.add_argument(
        "--recall-only",
        action="store_true",
        help=(
            "Skip pause/EOT points and write only recall points. "
            "Reuses an existing datapoint instead of wiping it"
        ),
    )
    parser.add_argument(
        "--curation",
        default="data/curation",
        help=(
            "Directory of curation decisions; applied when a source has one "
            "(default: data/curation)"
        ),
    )
    parser.add_argument(
        "--curate-only",
        action="store_true",
        help="Score and select split events into --curation, write no points",
    )
    parser.add_argument(
        "--judge-only",
        action="store_true",
        help=(
            "LLM-judge the gate survivors in --curation and re-rank selection, "
            "write no points"
        ),
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenAI model for --judge-only (default: {DEFAULT_JUDGE_MODEL})",
    )
    parser.add_argument(
        "--keep-per-type",
        type=int,
        default=DEFAULT_KEEP_PER_TYPE,
        help=(
            "Curation cap: pause/EOT events kept per datapoint "
            f"(default: {DEFAULT_KEEP_PER_TYPE})"
        ),
    )
    args = parser.parse_args()
    if args.judge_only:
        judge_curation(curation_dir=args.curation, model=args.judge_model)
        return
    if args.curate_only:
        run_curation(
            curation_dir=args.curation,
            datapoints=args.datapoint,
            profile=args.profile,
            datasource=args.datasource,
            limit=args.limit,
            keep_per_type=args.keep_per_type,
        )
        return
    try:
        require_whisper()
        if args.max_points is None or args.recall_only:
            require_tts()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    run(
        output_dir=args.output,
        datapoints=args.datapoint,
        profile=args.profile,
        datasource=args.datasource,
        limit=args.limit,
        max_points=args.max_points,
        max_datapoints=args.max_datapoints,
        recall_only=args.recall_only,
        curation_dir=args.curation,
    )


if __name__ == "__main__":
    main()
