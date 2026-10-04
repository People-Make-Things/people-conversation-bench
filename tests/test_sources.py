"""Tests for prepare-eval data-source adapters."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from bench.audio import TARGET_SAMPLE_RATE, read_wav_segment, write_wav_pcm16
from data_processing.audio import read_source_wav, write_eval_wav
from data_processing.ground_truth import (
    channel_words_from_ground_truth,
    labels_from_ground_truth,
)
from data_processing.labels import iter_split_events, iter_turns, load_labels
from data_processing.source_common import (
    download_s3_keys,
    list_s3_keys,
    list_s3_names,
)
from data_processing.source_seamless import SeamlessSource, parse_s3_uri
from data_processing.sources import get_source, resolve_datapoint


def ground_truth() -> dict:
    return {
        "events": [
            {"label": "voice_activation", "speaker": "A", "time": 0.5},
            {"label": "pause_start", "speaker": "A", "time": 1.0},
            {"label": "pause_end", "speaker": "A", "time": 1.4},
            {"label": "end_of_turn", "speaker": "A", "time": 2.0},
            {"label": "voice_activation", "speaker": "B", "time": 2.2},
            {"label": "end_of_turn", "speaker": "B", "time": 3.0},
            {"label": "backchannel_start", "speaker": "A", "time": 2.4},
            {"label": "backchannel_end", "speaker": "A", "time": 2.6},
        ],
        "words": [
            {"speaker": "A", "text": "hello", "start": 0.5, "end": 0.9},
            {"speaker": "B", "text": "there", "start": 2.2, "end": 2.8},
        ],
        "audio": {
            "A": "s3://pmt-data-seamless/raw/A/audio.wav",
            "B": "s3://pmt-data-seamless/raw/B/audio.wav",
        },
        "overlaps": [
            {
                "start": 2.2,
                "end": 2.4,
                "incumbent": "B",
                "entrant": "A",
            }
        ],
        "backchannels": [
            {"speaker": "A", "start": 2.4, "end": 2.6, "text": "mm"},
        ],
    }


def write_float_wav(
    path: Path,
    pcm: np.ndarray,
    sample_rate: int,
    extensible_fmt: bool = False,
) -> None:
    data = pcm.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, sample_rate, sample_rate * 4, 4, 32)
    extra = b"\x00\x00" if extensible_fmt else b""
    fact = b"fact" + struct.pack("<II", 4, pcm.shape[0]) if extensible_fmt else b""
    fmt_size = 16 + len(extra)
    riff_size = 4 + (8 + fmt_size) + len(fact) + (8 + len(data))
    with path.open("wb") as handle:
        handle.write(b"RIFF")
        handle.write(struct.pack("<I", riff_size))
        handle.write(b"WAVE")
        handle.write(b"fmt ")
        handle.write(struct.pack("<I", fmt_size))
        handle.write(fmt)
        handle.write(extra)
        handle.write(fact)
        handle.write(b"data")
        handle.write(struct.pack("<I", len(data)))
        handle.write(data)


def test_get_source_aliases() -> None:
    assert get_source("annotated").name == "annotated"
    assert get_source("pmt-data-annotated").name == "annotated"
    assert get_source("seamless").name == "seamless"
    assert get_source("pmt-data-seamless").name == "seamless"


def test_get_source_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown datasource"):
        get_source("missing")


def test_resolve_datapoint_normalizes_dashes() -> None:
    assert resolve_datapoint("example_1", ["example-1"]) == "example-1"


def test_parse_s3_uri() -> None:
    bucket, key = parse_s3_uri("s3://pmt-data-seamless/raw/A/audio.wav")
    assert bucket == "pmt-data-seamless"
    assert key == "raw/A/audio.wav"


def test_labels_from_ground_truth_maps_speakers_and_skips_backchannels() -> None:
    labels = labels_from_ground_truth(ground_truth())

    assert [event["type"] for event in labels["speaker_1"]] == [
        "voice_activation",
        "pause_start",
        "pause_end",
        "end_of_turn",
    ]
    assert [event["type"] for event in labels["speaker_2"]] == [
        "voice_activation",
        "end_of_turn",
    ]
    splits = iter_split_events(labels)
    assert [(event.split_type, event.active_speaker) for event in splits] == [
        ("pause_start", "speaker_1"),
        ("end_of_turn", "speaker_1"),
        ("end_of_turn", "speaker_2"),
    ]
    turns = iter_turns(labels, "speaker_1")
    assert turns[0].start == 0.5
    assert turns[0].end == 2.0
    assert labels["overlaps"] == [{"start": 2.2, "end": 2.4}]
    assert labels["backchannels"] == [
        {"speaker": "speaker_1", "start": 2.4, "end": 2.6, "text": "mm"}
    ]


def test_channel_words_from_ground_truth() -> None:
    words = channel_words_from_ground_truth(ground_truth())
    assert words["speaker_1"] == [{"text": "hello", "start": 0.5, "end": 0.9}]
    assert words["speaker_2"] == [{"text": "there", "start": 2.2, "end": 2.8}]


def test_read_source_wav_accepts_meta_style_float_wav(tmp_path: Path) -> None:
    source = tmp_path / "meta.wav"
    pcm = np.linspace(-0.2, 0.2, 4800, dtype=np.float32)
    write_float_wav(source, pcm, 48000, extensible_fmt=True)
    read_pcm, rate = read_source_wav(source)
    assert rate == 48000
    assert read_pcm.shape[0] == 4800


def test_write_eval_wav_converts_float32_48k_to_pcm16_24k(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    pcm = np.linspace(-0.2, 0.2, 48000, dtype=np.float32)
    write_float_wav(source, pcm, 48000)

    read_pcm, rate = read_source_wav(source)
    assert rate == 48000
    assert read_pcm.shape[0] == 48000

    dest = tmp_path / "speaker_1.wav"
    write_eval_wav(dest, read_pcm, rate)
    out, out_rate = read_wav_segment(dest)
    assert out_rate == TARGET_SAMPLE_RATE
    assert out.shape[0] == TARGET_SAMPLE_RATE


def test_read_source_wav_accepts_pcm16(tmp_path: Path) -> None:
    path = tmp_path / "pcm.wav"
    write_wav_pcm16(path, np.linspace(-0.1, 0.1, 2400, dtype=np.float32), 24000)
    pcm, rate = read_source_wav(path)
    assert rate == 24000
    assert pcm.shape[0] == 2400


class FakeS3:
    def __init__(self, files: dict[tuple[str, str], Path]) -> None:
        self.files = files

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        source = self.files[(bucket, key)]
        Path(filename).write_bytes(source.read_bytes())


def test_seamless_source_writes_canonical_datapoint(tmp_path: Path) -> None:
    payload = ground_truth()
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text(json.dumps(payload), encoding="utf-8")
    wav_a = tmp_path / "a.wav"
    wav_b = tmp_path / "b.wav"
    write_float_wav(wav_a, np.zeros(4800, dtype=np.float32), 48000)
    write_float_wav(wav_b, np.zeros(4800, dtype=np.float32), 48000)
    s3 = FakeS3(
        {
            ("pmt-data-seamless", "datasets/v1/sessions/seamless_demo/ground_truth.json"): gt_path,
            ("pmt-data-seamless", "raw/A/audio.wav"): wav_a,
            ("pmt-data-seamless", "raw/B/audio.wav"): wav_b,
        }
    )

    loaded = SeamlessSource().load(s3, "seamless_demo", tmp_path / "datapoint")

    assert loaded.source_id == "seamless_demo"
    assert loaded.source_labels.endswith("seamless_demo/ground_truth.json")
    assert loaded.source_audio["speaker_1"] == payload["audio"]["A"]
    labels = load_labels(tmp_path / "datapoint" / "label.json")
    assert iter_split_events(labels)[0].active_speaker == "speaker_1"
    assert (tmp_path / "datapoint" / "waves" / "speaker_1.wav").is_file()
    assert (tmp_path / "datapoint" / "waves" / "speaker_2.wav").is_file()
    assert not (tmp_path / "datapoint" / "waves" / "speaker_1_source.wav").exists()
    out, rate = read_wav_segment(tmp_path / "datapoint" / "waves" / "speaker_1.wav")
    assert rate == TARGET_SAMPLE_RATE
    assert loaded.channel_words is not None
    assert loaded.channel_words["speaker_1"][0]["text"] == "hello"


class ListingS3:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def get_paginator(self, name: str):
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str):
        contents = [
            {"Key": key, "Size": len(data)}
            for key, data in self.objects.items()
            if key.startswith(Prefix)
        ]
        yield {"Contents": contents}

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[key])


def test_list_and_download_s3_keys(tmp_path: Path) -> None:
    s3 = ListingS3(
        {
            "processed/src/points/000001/point.json": b"a",
            "processed/src/points/000002/point.json": b"b",
            "processed/other/points/000001/point.json": b"c",
            "processed/src/manifest.json": b"{}",
        }
    )

    assert list_s3_keys(s3, "bucket", "processed/src/") == [
        "processed/src/points/000001/point.json",
        "processed/src/points/000002/point.json",
        "processed/src/manifest.json",
    ]
    assert list_s3_names(s3, "bucket", "processed/", "/point.json") == [
        "other/points/000001",
        "src/points/000001",
        "src/points/000002",
    ]

    dest = tmp_path / "out"
    download_s3_keys(
        s3,
        "bucket",
        ["processed/src/points/000001/point.json"],
        dest,
        strip_prefix="processed/",
        workers=2,
    )
    assert (dest / "src/points/000001/point.json").read_bytes() == b"a"
    assert not (dest / "src/points/000002").exists()
