"""Load channel transcripts from annotation pipeline samples."""

from __future__ import annotations

import json
from pathlib import Path

SPEAKER_CHANNELS = {
    "speaker_1": "channel_1",
    "speaker_2": "channel_2",
}


def load_samples_row(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"{path}: expected at least one JSONL row")


def select_channel_words(channel: dict) -> list[dict]:
    best: list[dict] = []
    for transcript in channel.get("transcripts", []):
        words = transcript.get("words") or []
        if len(words) > len(best):
            best = words
    return best


def load_channel_words(samples_path: Path) -> dict[str, list[dict]]:
    row = load_samples_row(samples_path)
    words_by_channel = {
        channel["channel_id"]: select_channel_words(channel)
        for channel in row.get("channels", [])
    }
    return {
        speaker: words_by_channel[channel_id]
        for speaker, channel_id in SPEAKER_CHANNELS.items()
        if channel_id in words_by_channel
    }
