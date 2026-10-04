from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from bench.audio import (
    MODEL_FRAME_SAMPLES,
    PERSONAPLEX_CONTEXT_SEC,
    PERSONAPLEX_MAX_HISTORY_FRAMES,
    TARGET_SAMPLE_RATE,
    write_wav_pcm16,
)
from bench.rollout import EOT_HARD_CAP_SEC
from data_processing.audio import write_conversation_mix
from data_processing.cuts import (
    MAX_CHUNK_SEC,
    is_clean_cut,
    last_clean_cut,
    session_chunks,
)
from data_processing.labels import slice_labels, slice_words
from data_processing.point_audio import channel_timed_words
from data_processing.prepare_eval import (
    HISTORY_BUDGET_SEC,
    iter_session_chunks,
    write_chunk_dir,
)


def exchanging_labels() -> dict:
    return {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 31.0},
            {"type": "end_of_turn", "timestamp": 40.0},
            {"type": "voice_activation", "timestamp": 121.0},
            {"type": "end_of_turn", "timestamp": 130.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 20.0},
            {"type": "voice_activation", "timestamp": 40.0},
            {"type": "end_of_turn", "timestamp": 50.0},
        ],
        "merged": [],
        "overlaps": [{"start": 19.5, "end": 20.4}],
        "backchannels": [{"speaker": "speaker_1", "start": 12.0, "end": 12.4}],
    }


def test_history_budget_is_personaplex_window_minus_eot_cap() -> None:
    assert PERSONAPLEX_CONTEXT_SEC == pytest.approx(
        PERSONAPLEX_MAX_HISTORY_FRAMES * MODEL_FRAME_SAMPLES / TARGET_SAMPLE_RATE
    )
    assert HISTORY_BUDGET_SEC == pytest.approx(PERSONAPLEX_CONTEXT_SEC - EOT_HARD_CAP_SEC)
    assert MAX_CHUNK_SEC == pytest.approx(PERSONAPLEX_CONTEXT_SEC)


def test_handoff_is_clean_unless_an_overlap_covers_it() -> None:
    labels = exchanging_labels()
    assert is_clean_cut(labels, 30.0)
    assert is_clean_cut(labels, 20.0) is False
    assert is_clean_cut(labels, 40.0)
    assert is_clean_cut(labels, 35.0) is False


def test_last_clean_cut_stays_inside_slack() -> None:
    labels = exchanging_labels()
    assert last_clean_cut(labels, 30.0) == 30.0
    assert last_clean_cut(labels, 60.0) == 60.0
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 40.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 40.0},
            {"type": "end_of_turn", "timestamp": 55.0},
        ],
        "merged": [],
    }
    assert last_clean_cut(labels, 60.0, slack_sec=8.0) == 55.0
    assert last_clean_cut(labels, 90.0, slack_sec=8.0) is None


def test_session_chunks_pack_to_the_history_budget() -> None:
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 10.0},
            {"type": "end_of_turn", "timestamp": 20.0},
            {"type": "voice_activation", "timestamp": 130.0},
            {"type": "end_of_turn", "timestamp": 140.0},
            {"type": "voice_activation", "timestamp": 250.0},
            {"type": "end_of_turn", "timestamp": 260.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 25.0},
            {"type": "end_of_turn", "timestamp": 35.0},
            {"type": "voice_activation", "timestamp": 145.0},
            {"type": "end_of_turn", "timestamp": 155.0},
            {"type": "voice_activation", "timestamp": 265.0},
            {"type": "end_of_turn", "timestamp": 275.0},
        ],
        "merged": [],
    }
    chunks = session_chunks(labels, duration_sec=280.0, budget_sec=120.0)
    assert chunks[0][0] == 0.0
    assert chunks[0][1] == 120.0
    assert all(end - start <= 160.0 for start, end in chunks)
    for start, end in chunks[1:]:
        assert is_clean_cut(labels, start, duration_sec=280.0)
        assert is_clean_cut(labels, end, duration_sec=280.0)


def test_session_chunks_skip_windows_with_no_speech() -> None:
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 10.0},
            {"type": "voice_activation", "timestamp": 400.0},
            {"type": "end_of_turn", "timestamp": 410.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 12.0},
            {"type": "end_of_turn", "timestamp": 20.0},
            {"type": "voice_activation", "timestamp": 412.0},
            {"type": "end_of_turn", "timestamp": 420.0},
        ],
        "merged": [],
    }
    chunks = session_chunks(labels, duration_sec=430.0, budget_sec=120.0)
    assert chunks[0] == (0.0, 120.0)
    assert all(end > 400.0 or start < 30.0 for start, end in chunks)
    assert not any(120.0 <= start < 240.0 for start, _end in chunks)


def test_session_chunks_do_not_overshoot_the_personaplex_window() -> None:
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 10.0},
            {"type": "voice_activation", "timestamp": 180.0},
            {"type": "end_of_turn", "timestamp": 190.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 10.0},
            {"type": "end_of_turn", "timestamp": 180.0},
        ],
        "merged": [],
    }
    chunks = session_chunks(labels, duration_sec=200.0, budget_sec=120.0)
    assert all(end - start <= MAX_CHUNK_SEC + 1e-6 for start, end in chunks)
    assert not any(end - start > MAX_CHUNK_SEC for start, end in chunks)


def test_session_chunks_keep_an_unsplittable_session() -> None:
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 200.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 0.0},
            {"type": "end_of_turn", "timestamp": 200.0},
        ],
        "merged": [],
        "overlaps": [{"start": 0.0, "end": 200.0}],
    }
    chunks = session_chunks(labels, duration_sec=200.0)
    assert chunks[0][0] == 0.0
    assert chunks[0][1] == pytest.approx(MAX_CHUNK_SEC)
    assert all(end - start <= MAX_CHUNK_SEC + 1e-6 for start, end in chunks)
    assert chunks[-1][1] == pytest.approx(200.0)


def test_slice_labels_rebases_and_clips_busy_intervals() -> None:
    sliced = slice_labels(exchanging_labels(), 30.0, 50.0)
    assert sliced["speaker_1"][0]["timestamp"] == 1.0
    assert sliced["speaker_2"][0]["timestamp"] == 10.0
    assert sliced["overlaps"] == []
    assert sliced["backchannels"] == []


def test_slice_words_clips_to_the_window() -> None:
    words = slice_words(
        {
            "speaker_1": [{"text": "hi", "start": 29.0, "end": 31.0}],
            "speaker_2": [{"text": "no", "start": 60.0, "end": 61.0}],
        },
        30.0,
        50.0,
    )
    assert words is not None
    assert words["speaker_1"] == [{"text": "hi", "start": 0.0, "end": 1.0}]
    assert words["speaker_2"] == []


def test_channel_timed_words_rebases_onto_the_clip() -> None:
    words = channel_timed_words(
        {
            "speaker_1": [
                {"text": "hi", "start": 1.0, "end": 1.4},
                {"text": "there", "start": 2.0, "end": 2.5},
            ]
        },
        "speaker_1",
        1.2,
        2.2,
    )
    assert [word.text for word in words] == ["hi", "there"]
    assert words[0].start_sec == pytest.approx(0.0)
    assert words[0].end_sec == pytest.approx(0.2)
    assert words[1].start_sec == pytest.approx(0.8)
    assert words[1].end_sec == pytest.approx(1.0)


def test_write_conversation_mix_is_stereo_at_the_datapoint_root(tmp_path: Path) -> None:
    waves = tmp_path / "waves"
    waves.mkdir()
    write_wav_pcm16(waves / "speaker_1.wav", np.ones(2400, dtype=np.float32), 24000)
    write_wav_pcm16(waves / "speaker_2.wav", np.full(2400, 0.5, dtype=np.float32), 24000)
    dest = write_conversation_mix(tmp_path)
    assert dest == tmp_path / "conversation.wav"
    with wave.open(str(dest), "rb") as handle:
        assert handle.getnchannels() == 2
        assert handle.getnframes() == 2400


def test_write_chunk_dir_slices_audio_and_labels(tmp_path: Path) -> None:
    session = tmp_path / "session"
    waves = session / "waves"
    waves.mkdir(parents=True)
    write_wav_pcm16(waves / "speaker_1.wav", np.ones(24000 * 4, dtype=np.float32), 24000)
    write_wav_pcm16(waves / "speaker_2.wav", np.ones(24000 * 4, dtype=np.float32), 24000)
    (session / "label.json").write_text(
        json.dumps(
            {
                "speaker_1": [
                    {"type": "voice_activation", "timestamp": 0.0},
                    {"type": "end_of_turn", "timestamp": 1.5},
                ],
                "speaker_2": [
                    {"type": "voice_activation", "timestamp": 2.0},
                    {"type": "end_of_turn", "timestamp": 3.5},
                ],
                "merged": [
                    {
                        "id": "speaker_1",
                        "type": "end_of_turn",
                        "timestamp": 1.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    dest = tmp_path / "chunk"
    words = write_chunk_dir(
        session,
        dest,
        1.0,
        3.0,
        {"speaker_1": [{"text": "a", "start": 1.2, "end": 1.4}]},
    )
    labels = json.loads((dest / "label.json").read_text(encoding="utf-8"))
    assert labels["speaker_1"][0]["timestamp"] == 0.5
    assert words is not None
    assert words["speaker_1"][0]["start"] == pytest.approx(0.2)


def test_iter_session_chunks_names_multi_chunk_datapoints(tmp_path: Path) -> None:
    session = tmp_path / "session"
    waves = session / "waves"
    waves.mkdir(parents=True)
    duration = 280
    write_wav_pcm16(
        waves / "speaker_1.wav",
        np.zeros(24000 * duration, dtype=np.float32),
        24000,
    )
    write_wav_pcm16(
        waves / "speaker_2.wav",
        np.zeros(24000 * duration, dtype=np.float32),
        24000,
    )
    labels = {
        "speaker_1": [
            {"type": "voice_activation", "timestamp": 10.0},
            {"type": "end_of_turn", "timestamp": 20.0},
            {"type": "voice_activation", "timestamp": 130.0},
            {"type": "end_of_turn", "timestamp": 140.0},
            {"type": "voice_activation", "timestamp": 250.0},
            {"type": "end_of_turn", "timestamp": 260.0},
        ],
        "speaker_2": [
            {"type": "voice_activation", "timestamp": 25.0},
            {"type": "end_of_turn", "timestamp": 35.0},
            {"type": "voice_activation", "timestamp": 145.0},
            {"type": "end_of_turn", "timestamp": 155.0},
            {"type": "voice_activation", "timestamp": 265.0},
            {"type": "end_of_turn", "timestamp": 275.0},
        ],
        "merged": [],
    }
    (session / "label.json").write_text(json.dumps(labels), encoding="utf-8")
    chunks = iter_session_chunks(session, "demo_session", None, tmp_path / "out")
    assert len(chunks) >= 2
    assert chunks[0][0].name == "demo_session_000"
    assert chunks[0][2]["session_id"] == "demo_session"
    assert chunks[0][2]["chunk_index"] == 0
