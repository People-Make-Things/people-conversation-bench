"""Load Seamless Interaction sessions from pmt-data-seamless."""

from __future__ import annotations

import json
from pathlib import Path

from data_processing.audio import read_source_wav, write_eval_wav
from data_processing.ground_truth import (
    channel_words_from_ground_truth,
    labels_from_ground_truth,
)
from data_processing.source_common import LoadedDatapoint, list_s3_names

SEAMLESS_BUCKET = "pmt-data-seamless"
SESSIONS_PREFIX = "datasets/v1/sessions/"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected s3:// URI, got {uri!r}")
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    return bucket, key


class SeamlessSource:
    name = "seamless"

    def list_names(self, s3) -> list[str]:
        return list_s3_names(s3, SEAMLESS_BUCKET, SESSIONS_PREFIX, "/ground_truth.json")

    def load(self, s3, name: str, target: Path) -> LoadedDatapoint:
        target.mkdir(parents=True)
        labels_key = f"{SESSIONS_PREFIX}{name}/ground_truth.json"
        ground_truth_path = target / "ground_truth.json"
        s3.download_file(SEAMLESS_BUCKET, labels_key, str(ground_truth_path))
        with ground_truth_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)

        labels = labels_from_ground_truth(payload)
        with (target / "label.json").open("w", encoding="utf-8") as handle:
            json.dump(labels, handle, indent=2)
            handle.write("\n")

        waves_dir = target / "waves"
        waves_dir.mkdir()
        source_audio = {}
        audio = payload.get("audio") or {}
        for raw_speaker, speaker in (("A", "speaker_1"), ("B", "speaker_2")):
            uri = audio.get(raw_speaker)
            if not uri:
                raise ValueError(f"{name}: ground_truth missing audio.{raw_speaker}")
            bucket, key = parse_s3_uri(uri)
            raw_path = waves_dir / f"{speaker}_source.wav"
            s3.download_file(bucket, key, str(raw_path))
            pcm, sample_rate = read_source_wav(raw_path)
            write_eval_wav(waves_dir / f"{speaker}.wav", pcm, sample_rate)
            raw_path.unlink()
            source_audio[speaker] = uri

        words = channel_words_from_ground_truth(payload)
        return LoadedDatapoint(
            source_id=name.replace("-", "_"),
            source_labels=f"s3://{SEAMLESS_BUCKET}/{labels_key}",
            source_audio=source_audio,
            channel_words=words or None,
        )
