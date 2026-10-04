"""Tests for prepare-eval's root manifest and shared history cap."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bench.audio import MODEL_FRAME_SAMPLES, TARGET_SAMPLE_RATE, write_wav_pcm16
from data_processing.audio import wav_duration_sec
from data_processing.context import build_completed_context_events
from data_processing.labels import SplitEvent, history_start_sec
from data_processing.point_audio import write_duplex_history
from data_processing.prepare_eval import HISTORY_CAP_SEC, update_root_manifest


def split_fixture() -> tuple[dict, SplitEvent]:
    """A split at 320 s whose live turn starts at 300 s."""
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 50.0},
            {"type": "voice_activation", "timestamp": 210.0},
            {"type": "end_of_turn", "timestamp": 240.0},
            {"type": "voice_activation", "timestamp": 300.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 60.0},
            {"type": "end_of_turn", "timestamp": 200.0},
            {"type": "voice_activation", "timestamp": 250.0},
            {"type": "end_of_turn", "timestamp": 290.0},
        ],
        "merged": [],
    }
    event = SplitEvent(
        index=0,
        timestamp=320.0,
        split_type="end_of_turn",
        active_speaker="speaker_1",
        model_speaker="speaker_2",
    )
    return labels, event


def test_short_history_is_not_cut() -> None:
    labels, event = split_fixture()
    assert history_start_sec(labels, event, 300.0, cap_sec=400.0) == 0.0


def test_history_cut_snaps_to_a_turn_start() -> None:
    labels, event = split_fixture()
    # Raw cut would be 300 - 100 = 200; the first turn starting at or after
    # that is speaker_1's at 210.
    assert history_start_sec(labels, event, 300.0, cap_sec=100.0) == 210.0


def test_history_cut_without_boundary_falls_back_to_raw_cut() -> None:
    labels, event = split_fixture()
    # No turn starts inside [270, 300], so the cap still holds via a raw cut.
    assert history_start_sec(labels, event, 300.0, cap_sec=30.0) == 270.0


def test_default_cap_matches_personaplex_window() -> None:
    assert HISTORY_CAP_SEC == 163.84


def test_completed_context_drops_turns_before_the_cut() -> None:
    labels, event = split_fixture()
    channel_words = {
        "speaker_1": [
            {"text": "early", "start": 1.0, "end": 2.0},
            {"text": "recent", "start": 211.0, "end": 212.0},
        ],
        "speaker_2": [
            {"text": "old", "start": 61.0, "end": 62.0},
            {"text": "fresh", "start": 251.0, "end": 252.0},
        ],
    }

    events = build_completed_context_events(
        labels, channel_words, event, start_sec=210.0
    )

    texts = [item["item"]["content"][0]["text"] for item in events]
    assert texts == ["recent", "fresh"]


def test_duplex_history_starts_at_the_cut(tmp_path: Path) -> None:
    rate = TARGET_SAMPLE_RATE
    pcm = np.linspace(-0.5, 0.5, 10 * rate).astype(np.float32)
    user_src = tmp_path / "user_src.wav"
    assistant_src = tmp_path / "assistant_src.wav"
    write_wav_pcm16(user_src, pcm, rate)
    write_wav_pcm16(assistant_src, pcm, rate)

    write_duplex_history(
        tmp_path,
        user_src,
        assistant_src,
        end_sec=8.0,
        sample_rate=rate,
        transcribe=lambda path: [],
        start_sec=3.0,
    )

    user_len = wav_duration_sec(tmp_path / "rollout_user_context.wav")
    assistant_len = wav_duration_sec(tmp_path / "rollout_assistant.wav")
    assert user_len == assistant_len
    frame_sec = MODEL_FRAME_SAMPLES / rate
    assert 5.0 <= user_len < 5.0 + frame_sec
    assert (user_len * rate) % MODEL_FRAME_SAMPLES == 0


def read_points(output_root: Path) -> list[str]:
    with (output_root / "manifest.json").open(encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["point_count"] == len(payload["points"])
    return payload["points"]


def test_fresh_output_creates_new_manifest(tmp_path: Path) -> None:
    update_root_manifest(tmp_path, "session_a", ["points/000001/point.json"])
    assert read_points(tmp_path) == ["session_a/points/000001/point.json"]


def test_existing_output_keeps_other_sources(tmp_path: Path) -> None:
    update_root_manifest(tmp_path, "session_a", ["points/000001/point.json"])
    update_root_manifest(
        tmp_path,
        "session_b",
        ["points/000001/point.json", "points/000002/point.json"],
    )
    # Re-running session_a replaces only its own entries.
    update_root_manifest(tmp_path, "session_a", ["points/000002/point.json"])
    assert read_points(tmp_path) == [
        "session_b/points/000001/point.json",
        "session_b/points/000002/point.json",
        "session_a/points/000002/point.json",
    ]


def test_source_prefix_does_not_match_other_sources(tmp_path: Path) -> None:
    update_root_manifest(tmp_path, "session_a_long", ["points/000001/point.json"])
    update_root_manifest(tmp_path, "session_a", ["points/000001/point.json"])
    assert read_points(tmp_path) == [
        "session_a_long/points/000001/point.json",
        "session_a/points/000001/point.json",
    ]
