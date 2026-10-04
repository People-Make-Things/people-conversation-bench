"""Run the eval suite on Modal, one container per provider shard.

Local eval runs strain a laptop: hundreds of realtime websocket sessions plus
serialized Whisper scoring. Each shard container downloads the prepared points
from S3 (in-region, minutes), scores Whisper on a T4, and mirrors every
rollout to s3://pmt-data-seamless/results/<run_id>/ as it finishes, so
progress is observable from anywhere and a killed shard resumes from the
mirrored score.json files. Pull a run down with scripts/sync_results.py.

Shards mirror the provider limits found empirically: PersonaPlex and Moshi
share the Modal A100 grant so they run sequentially; the GPT Realtime
variants share one OpenAI account so they run sequentially; Gemini and Grok
are separate providers and share one container concurrently. Per-model
session concurrency stays in model.toml.

Run:   uv run modal run --detach bench/eval_modal.py --run-id <id>
Watch: uv run modal app logs people-bench-eval
Requires the pmt-aws, pmt-openai, pmt-model-keys, and hf-token Modal secrets.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import boto3
import modal

from bench.eval import model_dir_name, run_suite
from bench.results_s3 import ResultsMirror, is_rollout_score
from data_processing.prepare_modal import PROCESSED_PREFIX
from data_processing.source_common import download_s3_keys, list_s3_keys
from data_processing.source_seamless import SEAMLESS_BUCKET

app = modal.App("people-bench-eval")

# (models, sequential): sequential shards run one model's suite at a time.
SHARDS = (
    (("personaplex", "moshi"), True),
    (("gpt-realtime-fast", "gpt-realtime-slow", "gpt-realtime-semantic-low"), True),
    (("gemini-live", "grok-voice"), False),
)

DOWNLOAD_WORKERS = 32
DATASET_DIR = Path("/tmp/dataset")
RESULTS_DIR = Path("/tmp/results")

image = (
    modal.Image.debian_slim(python_version="3.11")
    # ffmpeg for Whisper decoding, libportaudio2 so the sounddevice import in
    # bench.interact (pulled in by bench.eval) binds.
    .apt_install("ffmpeg", "libportaudio2")
    .pip_install(
        "numpy>=1.26,<2.2",
        "boto3",
        "python-dotenv",
        "sphn>=0.1.4,<0.2",
        "websockets>=12.0",
        "sounddevice>=0.5.0",
        "openai-whisper",
        "tqdm",
        "matplotlib",
        "sentence-transformers>=5,<6",
    )
    # bench.registry resolves models/ next to bench/, i.e. /root/models here.
    .add_local_dir("models", "/root/models")
    .add_local_python_source("bench", "data_processing", "utils", "env")
)


def download_dataset(s3, manifest_key: str) -> Path:
    """Mirror the manifest's point directories from S3 to local disk."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = DATASET_DIR / "manifest.json"
    s3.download_file(SEAMLESS_BUCKET, manifest_key, str(manifest_path))
    with manifest_path.open(encoding="utf-8") as handle:
        points = json.load(handle)["points"]
    point_dirs = {rel.rsplit("/", 1)[0] + "/" for rel in points}
    keys = []
    for source_id in {rel.split("/", 1)[0] for rel in points}:
        prefix = f"{PROCESSED_PREFIX}{source_id}/"
        for key in list_s3_keys(s3, SEAMLESS_BUCKET, prefix):
            rel = key.removeprefix(PROCESSED_PREFIX)
            if rel.rsplit("/", 1)[0] + "/" in point_dirs:
                keys.append(key)
    download_s3_keys(
        s3,
        SEAMLESS_BUCKET,
        keys,
        DATASET_DIR,
        strip_prefix=PROCESSED_PREFIX,
        workers=DOWNLOAD_WORKERS,
    )
    print(f"dataset: {len(points)} points, {len(keys)} files", flush=True)
    return manifest_path


@app.function(
    image=image,
    gpu="T4",
    cpu=8,
    memory=16384,
    secrets=[
        modal.Secret.from_name("pmt-aws"),
        modal.Secret.from_name("pmt-openai"),
        modal.Secret.from_name("pmt-model-keys"),
        modal.Secret.from_name("hf-token"),
    ],
    timeout=24 * 60 * 60,
)
def eval_shard(
    models: tuple[str, ...],
    sequential: bool,
    manifest_key: str,
    run_id: str,
    rollouts: int,
    eval_type: str | None = None,
) -> dict:
    s3 = boto3.client("s3")
    manifest_path = download_dataset(s3, manifest_key)
    run_dir = RESULTS_DIR / run_id
    mirror = ResultsMirror(run_dir, client=s3)
    model_dirs = {model_dir_name(ref) for ref in models}
    restored = mirror.download(
        predicate=lambda rel: is_rollout_score(rel, model_dirs),
        workers=DOWNLOAD_WORKERS,
    )
    print(f"shard {list(models)}: {restored} scored rollout(s) restored", flush=True)

    groups = [[ref] for ref in models] if sequential else [list(models)]
    for group in groups:
        asyncio.run(
            run_suite(
                group,
                run_dir,
                manifest=manifest_path,
                rollouts=rollouts,
                eval_type=eval_type,
                mirror=mirror,
            )
        )
    scored = len(list(run_dir.glob("*/*/*/rollout_*/score.json")))
    return {"models": list(models), "scored": scored, "restored": restored}


@app.local_entrypoint()
def main(
    run_id: str,
    manifest: str = "data/processed/manifest_normal.json",
    rollouts: int = 2,
    profile: str = "pmt",
    models: str = "",
    eval_type: str = "",
) -> None:
    """models: comma-separated subset to run (default all shards).
    eval_type: run a single eval (e.g. pause-recognition) instead of the suite."""
    s3 = boto3.Session(profile_name=profile).client("s3")
    manifest_key = f"{PROCESSED_PREFIX}{Path(manifest).name}"
    s3.upload_file(manifest, SEAMLESS_BUCKET, manifest_key)
    print(f"manifest -> s3://{SEAMLESS_BUCKET}/{manifest_key}")

    selected = set(models.split(",")) if models else None
    if selected is not None:
        unknown = selected - {ref for shard_models, _ in SHARDS for ref in shard_models}
        if unknown:
            raise ValueError(f"unknown models: {sorted(unknown)}")
    shards = [
        (
            tuple(ref for ref in shard_models if selected is None or ref in selected),
            sequential,
        )
        for shard_models, sequential in SHARDS
    ]
    arguments = [
        (shard_models, sequential, manifest_key, run_id, rollouts, eval_type or None)
        for shard_models, sequential in shards
        if shard_models
    ]
    for result in eval_shard.starmap(arguments):
        print(f"shard done: {result}", flush=True)
    print(
        f"All shards finished. Sync locally with: "
        f"uv run python scripts/sync_results.py {run_id}"
    )
