"""Load reviewed annotations from pmt-data-annotated / pmt-data-raw."""

from __future__ import annotations

from pathlib import Path

from botocore.exceptions import ClientError

from data_processing.source_common import LoadedDatapoint, list_s3_names
from data_processing.transcripts import load_channel_words

ANNOTATION_BUCKET = "pmt-data-annotated"
RAW_BUCKET = "pmt-data-raw"
REVIEWED_PREFIX = "reviewed"
DRAFTS_PREFIX = "drafts"


class AnnotatedSource:
    name = "annotated"

    def list_names(self, s3) -> list[str]:
        return list_s3_names(s3, ANNOTATION_BUCKET, f"{REVIEWED_PREFIX}/", "/label.json")

    def load(self, s3, name: str, target: Path) -> LoadedDatapoint:
        target.mkdir(parents=True)
        labels_key = f"{REVIEWED_PREFIX}/{name}/label.json"
        s3.download_file(ANNOTATION_BUCKET, labels_key, str(target / "label.json"))

        waves_dir = target / "waves"
        waves_dir.mkdir()
        source_audio = {}
        for channel in (1, 2):
            speaker = f"speaker_{channel}"
            audio_key = f"{name}/channel_{channel}.wav"
            s3.download_file(RAW_BUCKET, audio_key, str(waves_dir / f"{speaker}.wav"))
            source_audio[speaker] = f"s3://{RAW_BUCKET}/{audio_key}"

        channel_words = None
        samples_key = f"{DRAFTS_PREFIX}/{name}/samples.jsonl"
        samples_path = target / "samples.jsonl"
        try:
            s3.download_file(ANNOTATION_BUCKET, samples_key, str(samples_path))
            channel_words = load_channel_words(samples_path)
        except ClientError:
            print(
                f"{name}: no draft transcripts at s3://{ANNOTATION_BUCKET}/{samples_key}, "
                "skipping text context",
                flush=True,
            )

        return LoadedDatapoint(
            source_id=name.replace("-", "_"),
            source_labels=f"s3://{ANNOTATION_BUCKET}/{labels_key}",
            source_audio=source_audio,
            channel_words=channel_words,
        )
