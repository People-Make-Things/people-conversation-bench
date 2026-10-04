"""Convert eval_labeling ground_truth.json into the bench label / word schema."""

from __future__ import annotations

SPEAKER_MAP = {
    "A": "speaker_1",
    "B": "speaker_2",
}

LABEL_TYPES = frozenset(
    {"voice_activation", "end_of_turn", "pause_start", "pause_end"}
)


def bench_speaker(raw: str) -> str:
    try:
        return SPEAKER_MAP[raw]
    except KeyError as exc:
        raise ValueError(f"unknown ground-truth speaker: {raw}") from exc


def labels_from_ground_truth(payload: dict) -> dict:
    speaker_events: dict[str, list[dict]] = {
        "speaker_1": [],
        "speaker_2": [],
    }
    merged: list[dict] = []
    for event in payload.get("events", []):
        label = event.get("label")
        if label not in LABEL_TYPES:
            continue
        speaker = bench_speaker(event["speaker"])
        item = {"type": label, "timestamp": float(event["time"])}
        speaker_events[speaker].append(item)
        merged.append({"id": speaker, **item})
    merged.sort(key=lambda event: event["timestamp"])
    overlaps = [
        {"start": float(item["start"]), "end": float(item["end"])}
        for item in payload.get("overlaps") or []
    ]
    backchannels = []
    for item in payload.get("backchannels") or []:
        copied = {
            "speaker": bench_speaker(item["speaker"]),
            "start": float(item["start"]),
            "end": float(item["end"]),
        }
        if item.get("text"):
            copied["text"] = item["text"]
        backchannels.append(copied)
    return {
        "speaker_1": speaker_events["speaker_1"],
        "speaker_2": speaker_events["speaker_2"],
        "merged": merged,
        "overlaps": overlaps,
        "backchannels": backchannels,
    }


def channel_words_from_ground_truth(payload: dict) -> dict[str, list[dict]]:
    words: dict[str, list[dict]] = {"speaker_1": [], "speaker_2": []}
    for word in payload.get("words", []):
        speaker = bench_speaker(word["speaker"])
        text = str(word.get("text") or word.get("word") or "").strip()
        if not text:
            continue
        words[speaker].append(
            {
                "text": text,
                "start": float(word["start"]),
                "end": float(word["end"]),
            }
        )
    if not words["speaker_1"] and not words["speaker_2"]:
        return {}
    return words
