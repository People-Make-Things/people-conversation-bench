"""Curate split events on Modal, one container per session.

Curation downloads two ~90 MB WAVs per session, which is bandwidth-bound
locally. Each Modal container sits next to S3, so the whole source list runs
in roughly the time one session takes locally.

Run: uv run modal run data_processing/curation_modal.py [--limit N]
Requires the pmt-aws Modal secret (AWS keys for the pmt profile).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import boto3
import modal

from data_processing.curation import (
    DEFAULT_KEEP_PER_TYPE,
    curate_datapoint,
    curation_payload,
    write_curation_payload,
)
from data_processing.curation_report import write_curation_figures
from data_processing.sources import get_source

app = modal.App("people-bench-curation")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy", "boto3", "matplotlib")
    .add_local_python_source("data_processing", "bench")
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("pmt-aws")],
    timeout=900,
    max_containers=128,
)
def curate_session(datasource: str, source_name: str, keep_per_type: int) -> dict:
    s3 = boto3.client("s3")
    source = get_source(datasource)
    with tempfile.TemporaryDirectory() as temp_dir:
        datapoint_dir = Path(temp_dir) / source_name.replace("-", "_")
        loaded = source.load(s3, source_name, datapoint_dir)
        candidates = curate_datapoint(
            datapoint_dir, loaded.channel_words, keep_per_type
        )
    return curation_payload(loaded.source_id, candidates, keep_per_type)


@app.local_entrypoint()
def main(
    curation_dir: str = "data/curation",
    datasource: str = "seamless",
    limit: int = 0,
    keep_per_type: int = DEFAULT_KEEP_PER_TYPE,
    profile: str = "pmt",
) -> None:
    output_path = Path(curation_dir)
    source = get_source(datasource)
    s3 = boto3.Session(profile_name=profile).client("s3")
    source_names = source.list_names(s3)
    if limit:
        source_names = source_names[:limit]
    print(f"Curating {len(source_names)} {source.name} session(s) on Modal")

    total = 0
    total_kept = 0
    arguments = [(datasource, name, keep_per_type) for name in source_names]
    for payload in curate_session.starmap(arguments):
        write_curation_payload(
            output_path / f"{payload['source_id']}.json", payload
        )
        kept = sum(1 for event in payload["events"] if event["keep"])
        total += len(payload["events"])
        total_kept += kept
        print(
            f"{payload['source_id']}: kept {kept}/{len(payload['events'])} "
            f"split events"
        )

    figure = write_curation_figures(output_path)
    print(
        f"Done. Kept {total_kept}/{total} split events across "
        f"{len(source_names)} session(s). Figures: {figure}"
    )
