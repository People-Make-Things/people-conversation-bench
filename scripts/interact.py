"""CLI wrapper for people-bench commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from bench import eval as eval_session
from bench import interact as interact_session
from bench import rollout
from bench.points import EVAL_TYPES
from bench.registry import discover_model_configs, load_model_config
from bench.report import analyze
from bench.report.summarize import write_run_summaries


def add_interact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        help="Model id (e.g. personaplex) or @path/to/model.toml",
    )
    parser.add_argument(
        "--voice-prompt",
        default=None,
        help="Override voice prompt from model.toml",
    )
    parser.add_argument(
        "--text-prompt",
        default=None,
        help="Override text prompt from model.toml",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (-1 to omit)",
    )
    parser.add_argument(
        "--preload-point",
        default=None,
        help="Eval point directory or point.json to preload before live conversation",
    )
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio input/output devices and exit",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime voice model benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)

    interact_parser = subparsers.add_parser(
        "interact",
        help="Live mic/speaker conversation with a model",
    )
    add_interact_arguments(interact_parser)

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run deterministic recorded-audio rollouts",
    )
    eval_parser.add_argument(
        "--model",
        nargs="+",
        required=True,
        help="Model id(s), or 'all'. Pass several to overlap them in one run.",
    )
    eval_input = eval_parser.add_mutually_exclusive_group(required=True)
    eval_input.add_argument(
        "--manifest",
        nargs="+",
        help="One or more manifests; points are concatenated",
    )
    eval_input.add_argument(
        "--point",
        help="Point directory or point.json to run without a manifest",
    )
    eval_parser.add_argument(
        "--output",
        default=None,
        help="Run directory (default: data/results/<timestamp>)",
    )
    eval_parser.add_argument("--rollouts", type=int, default=3)
    eval_parser.add_argument(
        "--eval-type",
        choices=EVAL_TYPES,
        default=None,
        help="Run one metric; omit to run all",
    )
    eval_parser.add_argument(
        "--pause-drain-sec",
        type=float,
        default=rollout.PAUSE_DRAIN_SEC,
        help="Silence streamed after the pause, which sets pause rollout length",
    )
    eval_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip spinning duplex GPU containers before the first rollout",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Flag suspect rollouts in a results directory",
    )
    analyze_parser.add_argument(
        "run_dir",
        help="Rollout directory or run directory under data/results",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Rebuild results.json and figures from stored scores",
    )
    report_parser.add_argument(
        "run_dir",
        help="Eval run directory containing results.json",
    )

    subparsers.add_parser(
        "models",
        help="List available models",
    )

    args = parser.parse_args()
    if args.command == "interact":
        interact_session.run(
            model=args.model,
            voice_prompt=args.voice_prompt,
            text_prompt=args.text_prompt,
            seed=args.seed,
            input_device=args.input_device,
            output_device=args.output_device,
            list_devices=args.list_devices,
            preload_point=args.preload_point,
        )
        return

    if args.command == "eval":
        eval_session.run(
            model_refs=args.model,
            manifest=args.manifest,
            point=args.point,
            output_dir=args.output,
            rollouts=args.rollouts,
            eval_type=args.eval_type,
            pause_drain_sec=args.pause_drain_sec,
            warmup=not args.no_warmup,
        )
        return

    if args.command == "analyze":
        analyze.run(args.run_dir)
        return

    if args.command == "report":
        _results, paths = write_run_summaries(
            Path(args.run_dir),
            manifest=None,
            point=None,
        )
        if not paths:
            print(f"No scores found in {args.run_dir}")
            return
        for path in paths:
            print(path)
        return

    if args.command == "models":
        configs = discover_model_configs()
        if not configs:
            print("No models found.")
            return
        for model_id in sorted(configs):
            model = load_model_config(configs[model_id])
            print(f"{model.id}\t{model.name}")
