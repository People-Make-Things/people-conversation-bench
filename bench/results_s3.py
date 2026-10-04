"""Mirror eval run artifacts to S3 as they are written.

Keys mirror the local layout: s3://pmt-data-seamless/results/<run_id>/...
matches data/results/<run_id>/... file for file, so a run can be pulled or
inspected remotely with the same paths `bench report` and `bench analyze`
expect locally.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import boto3

from data_processing.prepare_eval import AWS_PROFILE
from data_processing.source_common import download_s3_keys, list_s3_objects
from data_processing.source_seamless import SEAMLESS_BUCKET

RESULTS_PREFIX = "results/"
DOWNLOAD_WORKERS = 24

# Summary artifacts (run.json, results.json, figures) sit at most three path
# segments below the run dir; rollout artifacts start at
# <eval>/<model>/<point>/rollout_NNN/.
SUMMARY_MAX_DEPTH = 3
# eval / model / point / rollout_NNN / score.json
ROLLOUT_SCORE_PARTS = 5


def is_rollout_score(rel: str, models: set[str]) -> bool:
    parts = Path(rel).parts
    return (
        len(parts) == ROLLOUT_SCORE_PARTS
        and parts[-1] == "score.json"
        and parts[1] in models
    )


class ResultsMirror:
    def __init__(self, run_dir: Path, client=None, profile: str = AWS_PROFILE):
        self.run_dir = run_dir
        self.client = (
            client
            if client is not None
            else boto3.Session(profile_name=profile).client("s3")
        )

    def upload(self, paths: list[Path]) -> None:
        for path in paths:
            key = (
                f"{RESULTS_PREFIX}{self.run_dir.name}/"
                f"{path.relative_to(self.run_dir).as_posix()}"
            )
            self.client.upload_file(str(path), SEAMLESS_BUCKET, key)

    async def upload_dir(self, directory: Path) -> None:
        files = sorted(path for path in directory.rglob("*") if path.is_file())
        await asyncio.to_thread(self.upload, files)

    async def upload_summaries(self) -> None:
        files = sorted(
            path
            for path in self.run_dir.rglob("*")
            if path.is_file()
            and len(path.relative_to(self.run_dir).parts) <= SUMMARY_MAX_DEPTH
        )
        await asyncio.to_thread(self.upload, files)

    def download(
        self,
        *,
        predicate: Callable[[str], bool] | None = None,
        skip_same_size: bool = False,
        workers: int = DOWNLOAD_WORKERS,
    ) -> int:
        prefix = f"{RESULTS_PREFIX}{self.run_dir.name}/"
        keys = []
        for key, size in list_s3_objects(self.client, SEAMLESS_BUCKET, prefix):
            rel = key.removeprefix(prefix)
            if predicate is not None and not predicate(rel):
                continue
            dest = self.run_dir / rel
            if skip_same_size and dest.is_file() and dest.stat().st_size == size:
                continue
            keys.append(key)
        download_s3_keys(
            self.client,
            SEAMLESS_BUCKET,
            keys,
            self.run_dir,
            strip_prefix=prefix,
            workers=workers,
        )
        return len(keys)
