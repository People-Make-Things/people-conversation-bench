"""Tests for shared TimedWord helpers."""

from __future__ import annotations

import json
from pathlib import Path

from bench.protocol import TimedWord
from utils.timed_words import (
    load_timed_words,
    parse_timed_words,
    timed_words_document,
    timed_words_json_bytes,
    words_text,
    write_timed_words,
)


def test_parse_timed_words_skips_empty_text() -> None:
    words = parse_timed_words(
        [
            {"text": "hello", "start_sec": 0.0, "end_sec": 0.5},
            {"text": "  ", "start_sec": 1.0, "end_sec": 1.5},
        ]
    )
    assert len(words) == 1
    assert words[0].text == "hello"


def test_timed_words_round_trip_through_json(tmp_path: Path) -> None:
    words = (TimedWord(text="hi", start_sec=0.0, end_sec=0.4),)
    path = tmp_path / "assistant_transcript.json"
    write_timed_words(path, words)

    loaded = load_timed_words(path)
    assert loaded == words
    assert json.loads(timed_words_json_bytes(words).decode("utf-8")) == timed_words_document(words)


def test_words_text_joins_stripped_words() -> None:
    assert (
        words_text(
            (
                TimedWord(text=" Hello", start_sec=0.0, end_sec=0.3),
                TimedWord(text="world", start_sec=0.3, end_sec=0.6),
            )
        )
        == "Hello world"
    )
    assert words_text(()) == ""
