"""Tests for PersonaPlex wire protocol helpers."""

from __future__ import annotations

from bench.audio import MODEL_FRAME_SAMPLES
from models.personaplex.server.wire import (
    HISTORY_CHUNK_BYTES,
    MODEL_FRAME_BYTES,
    build_meta_payload,
    build_transcript_payload,
    iter_byte_chunks,
)
from bench.protocol import TimedWord


def test_wire_frame_bytes_align_to_model_frames() -> None:
    assert MODEL_FRAME_BYTES == MODEL_FRAME_SAMPLES * 2
    assert HISTORY_CHUNK_BYTES % MODEL_FRAME_BYTES == 0


def test_iter_byte_chunks_preserves_payload() -> None:
    payload = bytes(range(256))
    chunks = list(iter_byte_chunks(payload, chunk_size=MODEL_FRAME_BYTES * 4))
    assert b"".join(chunks) == payload


def test_build_meta_and_transcript_payloads() -> None:
    meta = build_meta_payload("NATF2.pt", "hello", 42)
    assert b"voice_prompt" in meta
    transcript = build_transcript_payload(
        (TimedWord(text="hi", start_sec=0.0, end_sec=0.5),)
    )
    assert b'"words"' in transcript
