"""Shared types and S3 listing for prepare-eval source adapters."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDatapoint:
    source_id: str
    source_labels: str
    source_audio: dict[str, str]
    channel_words: dict[str, list[dict]] | None


def list_s3_objects(s3, bucket: str, prefix: str) -> list[tuple[str, int]]:
    items = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            items.append((item["Key"], int(item["Size"])))
    return items


def list_s3_keys(s3, bucket: str, prefix: str) -> list[str]:
    return [key for key, _size in list_s3_objects(s3, bucket, prefix)]


def list_s3_names(s3, bucket: str, prefix: str, suffix: str) -> list[str]:
    return sorted(
        key[len(prefix) : -len(suffix)]
        for key in list_s3_keys(s3, bucket, prefix)
        if key.endswith(suffix)
    )


def download_s3_keys(
    s3,
    bucket: str,
    keys: Sequence[str],
    dest_root: Path,
    *,
    strip_prefix: str,
    workers: int,
) -> None:
    def fetch(key: str) -> None:
        dest = dest_root / key.removeprefix(strip_prefix)
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, key, str(dest))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, keys))
