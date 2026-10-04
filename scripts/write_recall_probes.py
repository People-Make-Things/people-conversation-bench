"""Write listen-test reference + clone WAVs for a data source."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import boto3

from bench.audio import write_wav_pcm16
from data_processing.labels import load_labels
from data_processing.personas import DEFAULT_PERSONA_MODEL, persona_model_name
from data_processing.recall import (
    load_reference_clip,
    reference_turns,
    speaker_turn_text,
    style_probe,
    styled_recall_probes,
)
from data_processing.sources import DEFAULT_DATASOURCE, get_source, resolve_datapoint
from data_processing.tts import synthesize_clone
from utils.openai import openai_json_complete

WAIT_WHAT = "What were we just talking about?"


def write_speaker_probes(
    out_dir: Path,
    *,
    source_id: str,
    speaker: str,
    labels: dict,
    speaker_path: Path,
    channel_words: dict[str, list[dict]] | None,
    complete,
) -> dict | None:
    turns = reference_turns(labels, speaker)
    if turns is None:
        print(f"{source_id} {speaker}: not enough reference speech, skipping", flush=True)
        return None

    reference_pcm, reference_rate, reference_text = load_reference_clip(
        speaker_path, turns, channel_words, speaker
    )
    speaker_dir = out_dir / speaker
    speaker_dir.mkdir(parents=True, exist_ok=True)
    write_wav_pcm16(speaker_dir / "reference.wav", reference_pcm, reference_rate)

    filenames = {
        "fact_recall": "fact.wav",
        "conversation_recall": "conversation.wav",
    }
    probes = [
        (
            "wait_what.wav",
            style_probe(
                WAIT_WHAT,
                speaker_turn_text(labels, channel_words, speaker),
                complete=complete,
            ),
            None,
        )
    ]
    for action, spoken, gold in styled_recall_probes(
        source_id, labels, channel_words, speaker, complete
    ):
        probes.append((filenames[action], spoken, gold))

    written = []
    for filename, spoken, gold in probes:
        print(f"{source_id} {speaker}: cloning {filename}", flush=True)
        pcm = synthesize_clone(
            reference_pcm, reference_rate, reference_text, spoken
        )
        write_wav_pcm16(speaker_dir / filename, pcm, reference_rate)
        entry = {"file": filename, "text": spoken}
        if gold is not None:
            entry["gold_answer"] = gold
        written.append(entry)

    return {
        "reference_sec": float(reference_pcm.shape[0] / reference_rate),
        "reference_text": reference_text,
        "probes": written,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write reference + TTS clone WAVs for listening",
    )
    parser.add_argument("datapoint", nargs="+")
    parser.add_argument("--datasource", default=DEFAULT_DATASOURCE)
    parser.add_argument("--output", default="data/analysis/recall_probes")
    parser.add_argument("--profile", default="pmt")
    args = parser.parse_args()

    source = get_source(args.datasource)
    s3 = boto3.Session(profile_name=args.profile).client("s3")
    available = source.list_names(s3)
    names = [resolve_datapoint(name, available) for name in args.datapoint]
    model_name = persona_model_name() or DEFAULT_PERSONA_MODEL
    complete = lambda system, prompt: openai_json_complete(
        system, prompt, model=model_name
    )

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        for name in names:
            datapoint_dir = Path(temp_dir) / name.replace("-", "_")
            loaded = source.load(s3, name, datapoint_dir)
            labels = load_labels(datapoint_dir / "label.json")
            out_dir = output_root / loaded.source_id
            speakers = {}
            for speaker in ("speaker_1", "speaker_2"):
                payload = write_speaker_probes(
                    out_dir,
                    source_id=loaded.source_id,
                    speaker=speaker,
                    labels=labels,
                    speaker_path=datapoint_dir / "waves" / f"{speaker}.wav",
                    channel_words=loaded.channel_words,
                    complete=complete,
                )
                if payload is not None:
                    speakers[speaker] = payload
            manifest = {
                "source_id": loaded.source_id,
                "datasource": source.name,
                "source_labels": loaded.source_labels,
                "source_audio": loaded.source_audio,
                "speakers": speakers,
            }
            with (out_dir / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
                handle.write("\n")
            print(f"{loaded.source_id}: wrote probes -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
