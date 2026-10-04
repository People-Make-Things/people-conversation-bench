"""Wire protocol for PersonaPlex duplex preload uploads."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence

from bench.audio import MODEL_FRAME_SAMPLES
from bench.protocol import TimedWord
from utils.timed_words import timed_words_json_bytes

MSG_HANDSHAKE = 0x00
MSG_AUDIO = 0x01
MSG_TEXT = 0x02
MSG_META = 0x03
MSG_USER_HISTORY = 0x04
MSG_ASSISTANT_HISTORY = 0x05
MSG_TRANSCRIPT = 0x06
MSG_COMMIT = 0x07
MSG_PRIMED = 0x08

MODEL_FRAME_BYTES = MODEL_FRAME_SAMPLES * 2
HISTORY_CHUNK_BYTES = MODEL_FRAME_BYTES * 256


def iter_byte_chunks(data: bytes, chunk_size: int = HISTORY_CHUNK_BYTES) -> Iterator[bytes]:
    if chunk_size % MODEL_FRAME_BYTES != 0:
        raise ValueError("chunk_size must be aligned to model frame bytes")
    for offset in range(0, len(data), chunk_size):
        yield data[offset : offset + chunk_size]


def build_meta_payload(
    voice_prompt: str,
    text_prompt: str,
    seed: int | None,
) -> bytes:
    payload = {
        "voice_prompt": voice_prompt,
        "text_prompt": text_prompt,
    }
    if seed is not None:
        payload["seed"] = seed
    return json.dumps(payload).encode("utf-8")


def build_transcript_payload(words: Sequence[TimedWord]) -> bytes:
    return timed_words_json_bytes(words)
