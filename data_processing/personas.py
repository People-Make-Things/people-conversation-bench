"""Generate identity-only system prompts from a conversation transcript."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import env
from data_processing.context import extract_turn_text
from data_processing.labels import iter_turns
from utils.openai import openai_json_complete

DEFAULT_PERSONA_MODEL = "gpt-4o"
PROMPT_VERSION = 2
SPEAKERS = ("speaker_1", "speaker_2")

PERSONA_FRAME = (
    "You are continuing this live spoken conversation as this person. "
    "Assistant messages in the history are your previous speech. "
    "Stay in character. Do not mention being an AI."
)

GENERATOR_SYSTEM = """\
You write short identity briefs for two people in a conversation transcript.

Return JSON with keys speaker_1 and speaker_2. Each value is a plain string \
holding a second-person persona of 2-6 sentences of prose, never a nested \
object and never a field-by-field breakdown.

Cover only what is known, in prose:
- who they are (name, origin, occupation) if they stated it about themselves
- their relationship to the other speaker
- speaking style (formality, pace, fillers)

Do not include:
- answers to questions asked in the conversation
- a recap of what was discussed
- what they should say next
- facts that belong only to the other speaker
- any instruction to mention being an AI

If little is known, keep it to style and relationship."""


def format_conversation_transcript(
    labels: dict,
    channel_words: dict[str, list[dict]],
) -> str:
    turns: list[tuple[float, str, str]] = []
    for speaker in SPEAKERS:
        for turn in iter_turns(labels, speaker):
            text = extract_turn_text(
                channel_words.get(speaker, []),
                turn.start,
                turn.end,
            ).strip()
            if text:
                turns.append((turn.start, speaker, text))
    turns.sort(key=lambda item: item[0])
    return "\n".join(f"{speaker}: {text}" for _, speaker, text in turns)


def compose_text_prompt(persona: str) -> str:
    persona = persona.strip()
    if not persona:
        return PERSONA_FRAME
    return f"{PERSONA_FRAME}\n\n{persona}"


def parse_persona_response(text: str) -> dict[str, str]:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("persona response must be a JSON object")
    personas: dict[str, str] = {}
    for speaker in SPEAKERS:
        value = payload.get(speaker)
        if not isinstance(value, str):
            raise ValueError(f"persona response missing string {speaker}")
        personas[speaker] = value.strip()
    return personas


def openai_complete(prompt: str, *, model: str) -> str:
    return openai_json_complete(GENERATOR_SYSTEM, prompt, model=model)


def build_personas_payload(
    speaker_personas: dict[str, str],
    *,
    model: str | None,
    transcript_sha256: str | None,
) -> dict:
    return {
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "transcript_sha256": transcript_sha256,
        **{
            speaker: {
                "persona": speaker_personas[speaker],
                "text_prompt": compose_text_prompt(speaker_personas[speaker]),
            }
            for speaker in SPEAKERS
        },
    }


def frame_only_personas() -> dict:
    return build_personas_payload(
        {speaker: "" for speaker in SPEAKERS},
        model=None,
        transcript_sha256=None,
    )


def persona_model_name(override: str | None = None) -> str:
    if override:
        return override
    return env.get("PERSONA_MODEL") or DEFAULT_PERSONA_MODEL


def generate_personas(
    labels: dict,
    channel_words: dict[str, list[dict]],
    *,
    model: str | None = None,
    complete: Callable[[str], str] | None = None,
) -> dict:
    transcript = format_conversation_transcript(labels, channel_words)
    if not transcript:
        return frame_only_personas()

    resolved_model = persona_model_name(model)
    completer = complete or (
        lambda prompt: openai_complete(prompt, model=resolved_model)
    )
    parsed = parse_persona_response(
        completer(f"Transcript:\n{transcript}"),
    )
    digest = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    return build_personas_payload(
        parsed,
        model=resolved_model,
        transcript_sha256=digest,
    )


def write_personas(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
