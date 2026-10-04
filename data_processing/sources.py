"""Registry of prepare-eval data sources.

Each source loads conversations into the same on-disk layout process_datapoint
already expects: label.json plus waves/speaker_{1,2}.wav.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from data_processing.source_annotated import AnnotatedSource
from data_processing.source_common import LoadedDatapoint
from data_processing.source_seamless import SeamlessSource

DEFAULT_DATASOURCE = "annotated"

SOURCE_ALIASES = {
    "annotated": "annotated",
    "pmt-data-annotated": "annotated",
    "seamless": "seamless",
    "pmt-data-seamless": "seamless",
}


class DataSource(Protocol):
    name: str

    def list_names(self, s3) -> list[str]: ...

    def load(self, s3, name: str, target: Path) -> LoadedDatapoint: ...


SOURCES: dict[str, DataSource] = {
    "annotated": AnnotatedSource(),
    "seamless": SeamlessSource(),
}


def get_source(name: str) -> DataSource:
    key = SOURCE_ALIASES.get(name)
    if key is None:
        known = ", ".join(sorted(set(SOURCE_ALIASES)))
        raise ValueError(f"unknown datasource {name!r}; expected one of: {known}")
    return SOURCES[key]


def resolve_datapoint(name: str, available: list[str]) -> str:
    if name in available:
        return name
    normalized = name.replace("-", "_")
    matches = [item for item in available if item.replace("-", "_") == normalized]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"datapoint not found: {name}")
