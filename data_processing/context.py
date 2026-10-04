"""Build conversation context events for eval point prefill."""

from __future__ import annotations

import json
from pathlib import Path

from bench.audio import pcm_float_to_b64
from data_processing.audio import read_wav_range
from data_processing.labels import (
    SplitEvent,
    current_turn_start,
    iter_completed_turns,
    iter_prefix_turns,
)


def extract_turn_text(words: list[dict], start_sec: float, end_sec: float) -> str:
    selected = [
        word
        for word in words
        if float(word["end"]) > start_sec and float(word["start"]) < end_sec
    ]
    selected.sort(key=lambda word: float(word["start"]))
    return " ".join(str(word.get("text", "")).strip() for word in selected if word.get("text"))


def speaker_role(speaker: str, user_speaker: str) -> str:
    return "user" if speaker == user_speaker else "assistant"


def text_message_event(role: str, text: str) -> dict:
    if role == "user":
        content_type = "input_text"
    else:
        content_type = "output_text"
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": role,
            "content": [{"type": content_type, "text": text}],
        },
    }


def audio_message_event(audio_b64: str, transcript: str | None = None) -> dict:
    content: dict = {"type": "input_audio", "audio": audio_b64}
    if transcript:
        content["transcript"] = transcript
    return {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [content],
        },
    }


def build_context_events(
    labels: dict,
    channel_words: dict[str, list[dict]],
    speaker_paths: dict[str, Path],
    split_event: SplitEvent,
    input_end_sec: float,
) -> list[dict]:
    user_speaker = split_event.active_speaker
    events: list[dict] = []

    for turn in iter_completed_turns(labels, split_event):
        role = speaker_role(turn.speaker, user_speaker)
        text = extract_turn_text(channel_words.get(turn.speaker, []), turn.start, turn.end)
        if text.strip():
            events.append(text_message_event(role, text))

    turn_start = current_turn_start(labels, user_speaker, split_event.timestamp)
    user_pcm, _ = read_wav_range(
        speaker_paths[user_speaker],
        turn_start,
        input_end_sec,
    )
    transcript = extract_turn_text(
        channel_words.get(user_speaker, []),
        turn_start,
        input_end_sec,
    )
    events.append(
        audio_message_event(
            pcm_float_to_b64(user_pcm),
            transcript or None,
        )
    )
    return events


def build_completed_context_events(
    labels: dict,
    channel_words: dict[str, list[dict]],
    split_event: SplitEvent,
    start_sec: float = 0.0,
) -> list[dict]:
    events = []
    for turn in iter_completed_turns(labels, split_event):
        if turn.start < start_sec - 1e-6:
            continue
        role = speaker_role(turn.speaker, split_event.active_speaker)
        text = extract_turn_text(channel_words.get(turn.speaker, []), turn.start, turn.end)
        if text.strip():
            events.append(text_message_event(role, text))
    return events


def build_prefix_context_events(
    labels: dict,
    channel_words: dict[str, list[dict]],
    user_speaker: str,
    end_sec: float,
) -> list[dict]:
    events = []
    for turn in iter_prefix_turns(labels, end_sec):
        role = speaker_role(turn.speaker, user_speaker)
        text = extract_turn_text(
            channel_words.get(turn.speaker, []),
            turn.start,
            turn.end,
        )
        if text.strip():
            events.append(text_message_event(role, text))
    return events


def write_context(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"events": events}, handle, indent=2)
        handle.write("\n")
