"""Shared TimedWord parsing and transcript JSON helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from bench.protocol import TimedWord


def parse_timed_words(entries: Sequence[Mapping[str, object]]) -> tuple[TimedWord, ...]:
    return tuple(
        word
        for entry in entries
        if (word := TimedWord.from_dict(entry)).text
    )


def load_timed_words(path: Path) -> tuple[TimedWord, ...]:
    if not path.is_file():
        return ()
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    words = payload.get("words", payload)
    if not isinstance(words, list):
        raise ValueError(f"{path}: expected a list of words")
    return parse_timed_words(words)


def timed_words_document(words: Sequence[TimedWord]) -> dict:
    return {"words": [word.to_dict() for word in words]}


def timed_words_json_bytes(words: Sequence[TimedWord]) -> bytes:
    return json.dumps(timed_words_document(words)).encode("utf-8")


def write_timed_words(path: Path, words: Sequence[TimedWord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(timed_words_document(words), handle, indent=2)
        handle.write("\n")


def words_text(words: Sequence[TimedWord]) -> str:
    return " ".join(word.text.strip() for word in words if word.text.strip())
