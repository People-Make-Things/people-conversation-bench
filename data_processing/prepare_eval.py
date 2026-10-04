"""Prepare role-separated conversation context for evaluation."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import boto3

from bench.audio import (
    PERSONAPLEX_CONTEXT_SEC,
    pad_pcm_to_model_frames,
    read_wav_segment,
    write_wav_pcm16,
)
from bench.points import RECALL_ACTIONS, ReferenceResponse
from bench.rollout import EOT_HARD_CAP_SEC
from data_processing.audio import read_wav_range, wav_duration_sec, write_conversation_mix
from data_processing.context import (
    build_completed_context_events,
    build_context_events,
    write_context,
)
from data_processing.curation import load_curation, skip_reason
from data_processing.cuts import session_chunks
from data_processing.labels import (
    RolloutWindow,
    SplitEvent,
    history_start_sec,
    iter_split_events,
    load_labels,
    next_turn_after,
    rollout_window,
    slice_labels,
    slice_words,
)
from data_processing.personas import (
    frame_only_personas,
    generate_personas,
    write_personas,
)
from data_processing.point_audio import (
    transcript_for,
    write_duplex_history,
    write_reference_response,
    write_rollout_input_audio,
)
from data_processing.recall import write_recall_points
from data_processing.sources import DEFAULT_DATASOURCE, get_source, resolve_datapoint
from utils.timed_words import write_timed_words

AWS_PROFILE = "pmt"
HISTORY_BUDGET_SEC = PERSONAPLEX_CONTEXT_SEC - EOT_HARD_CAP_SEC

# Shared history budget for pause / EOT points: PersonaPlex's trained window,
# 2048 mimi frames at 80 ms (see models/personaplex/server/prefill.py). Baking
# the cap into the artifacts means every model conditions on exactly the audio
# and text that sit on disk — the server-side truncation never fires — so an
# inspected point is the point every model was measured on.
HISTORY_CAP_SEC = 2048 * 0.08


def speaker_audio_path(datapoint_dir: Path, speaker: str) -> Path:
    return datapoint_dir / "waves" / f"{speaker}.wav"


def write_split_point(
    point_dir: Path,
    source_id: str,
    event: SplitEvent,
    labels: dict,
    speaker_paths: dict[str, Path],
    personas_payload: dict,
    channel_words: dict[str, list[dict]] | None,
    point_index: int,
    curation_decision: dict | None = None,
) -> str:
    window = rollout_window(labels, event)
    reference_turn = None
    if event.split_type == "end_of_turn":
        reference_turn = next_turn_after(
            labels, event.model_speaker, event.timestamp
        )
        if reference_turn is None:
            raise ValueError("eot points require a following assistant turn")

    user_pcm, sample_rate = read_wav_segment(
        speaker_paths[event.active_speaker], window.input_end
    )
    assistant_pcm, _ = read_wav_segment(
        speaker_paths[event.model_speaker], window.input_end
    )
    user_pcm = pad_pcm_to_model_frames(user_pcm)
    assistant_pcm = pad_pcm_to_model_frames(assistant_pcm)
    user_path = point_dir / "user.wav"
    assistant_path = point_dir / "assistant.wav"
    write_wav_pcm16(user_path, user_pcm, sample_rate)
    write_wav_pcm16(assistant_path, assistant_pcm, sample_rate)

    write_timed_words(
        point_dir / "assistant_transcript.json",
        transcript_for(
            channel_words, event.model_speaker, 0.0, window.input_end
        )(assistant_path),
    )

    if channel_words is not None:
        write_context(
            point_dir / "context.json",
            build_context_events(
                labels,
                channel_words,
                speaker_paths,
                event,
                input_end_sec=window.input_end,
            ),
        )

    history_start = history_start_sec(
        labels, event, window.turn_start, HISTORY_CAP_SEC
    )
    write_duplex_history(
        point_dir,
        speaker_paths[event.active_speaker],
        speaker_paths[event.model_speaker],
        window.turn_start,
        sample_rate,
        transcribe=transcript_for(
            channel_words, event.model_speaker, history_start, window.turn_start
        ),
        start_sec=history_start,
    )
    input_audio = write_rollout_input_audio(
        point_dir,
        speaker_paths[event.active_speaker],
        event,
        window,
        sample_rate,
    )
    reference = None
    if reference_turn is not None:
        reference = write_reference_response(
            point_dir,
            speaker_paths[event.model_speaker],
            reference_turn,
            sample_rate,
            transcribe=transcript_for(
                channel_words,
                event.model_speaker,
                reference_turn.start,
                reference_turn.end,
            ),
        )
    if channel_words is not None:
        write_context(
            point_dir / "rollout_context.json",
            build_completed_context_events(
                labels, channel_words, event, start_sec=history_start
            ),
        )

    point_id = f"{source_id}_{point_index:06d}"
    point_meta = build_point_meta(
        point_id,
        source_id,
        event,
        window,
        sample_rate,
        input_audio,
        personas_payload[event.model_speaker]["text_prompt"],
        include_context=channel_words is not None,
        reference=reference,
        history_start=history_start,
    )
    if curation_decision is not None:
        point_meta["curation"] = {
            "score": curation_decision["score"],
            "features": curation_decision["features"],
        }
    with (point_dir / "point.json").open("w", encoding="utf-8") as handle:
        json.dump(point_meta, handle, indent=2)
        handle.write("\n")
    return f"points/{point_index:06d}/point.json"


def write_chunk_dir(
    session_dir: Path,
    dest: Path,
    start_sec: float,
    end_sec: float,
    channel_words: dict[str, list[dict]] | None,
) -> dict[str, list[dict]] | None:
    dest.mkdir(parents=True)
    labels = load_labels(session_dir / "label.json")
    with (dest / "label.json").open("w", encoding="utf-8") as handle:
        json.dump(slice_labels(labels, start_sec, end_sec), handle, indent=2)
        handle.write("\n")
    waves_dir = dest / "waves"
    waves_dir.mkdir()
    for speaker in ("speaker_1", "speaker_2"):
        pcm, sample_rate = read_wav_range(
            speaker_audio_path(session_dir, speaker),
            start_sec,
            end_sec,
        )
        write_wav_pcm16(waves_dir / f"{speaker}.wav", pcm, sample_rate)
    return slice_words(channel_words, start_sec, end_sec)


def iter_session_chunks(
    session_dir: Path,
    session_id: str,
    channel_words: dict[str, list[dict]] | None,
    temp_dir: Path,
) -> list[tuple[Path, dict[str, list[dict]] | None, dict]]:
    labels = load_labels(session_dir / "label.json")
    duration_sec = min(
        wav_duration_sec(speaker_audio_path(session_dir, speaker))
        for speaker in ("speaker_1", "speaker_2")
    )
    chunks = session_chunks(
        labels,
        duration_sec=duration_sec,
        budget_sec=HISTORY_BUDGET_SEC,
    )
    if not chunks:
        chunks = [(0.0, duration_sec)]
    multi = len(chunks) > 1
    written: list[tuple[Path, dict[str, list[dict]] | None, dict]] = []
    for index, (start_sec, end_sec) in enumerate(chunks):
        source_id = f"{session_id}_{index:03d}" if multi else session_id
        dest = temp_dir / source_id
        words = write_chunk_dir(
            session_dir, dest, start_sec, end_sec, channel_words
        )
        written.append(
            (
                dest,
                words,
                {
                    "session_id": session_id,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "source_start_sec": start_sec,
                    "source_end_sec": end_sec,
                    "history_budget_sec": HISTORY_BUDGET_SEC,
                },
            )
        )
    return written


def load_personas(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def recall_rel_paths(output_dir: Path, point_paths: list[str]) -> list[str]:
    recalled = []
    for rel in point_paths:
        action = json.loads((output_dir / rel).read_text(encoding="utf-8")).get(
            "rollout", {}
        ).get("expected_action")
        if action in RECALL_ACTIONS:
            recalled.append(rel)
    return recalled


def write_recall_onto_existing(
    output_dir: Path,
    channel_words: dict[str, list[dict]] | None,
    chunk: dict | None,
) -> int:
    labels = load_labels(output_dir / "label.json")
    speaker_paths = {
        speaker: speaker_audio_path(output_dir, speaker)
        for speaker in ("speaker_1", "speaker_2")
    }
    for path in speaker_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"{output_dir.name}: missing {path}")
    personas_path = output_dir / "personas.json"
    if not personas_path.is_file():
        raise FileNotFoundError(f"{output_dir.name}: missing personas.json")
    if chunk is not None:
        existing_chunk = json.loads((output_dir / "chunk.json").read_text(encoding="utf-8"))
        if (
            abs(existing_chunk["source_start_sec"] - chunk["source_start_sec"]) > 1e-3
            or abs(existing_chunk["source_end_sec"] - chunk["source_end_sec"]) > 1e-3
        ):
            raise ValueError(
                f"{output_dir.name}: existing chunk "
                f"{existing_chunk['source_start_sec']}-"
                f"{existing_chunk['source_end_sec']} does not match "
                f"{chunk['source_start_sec']}-{chunk['source_end_sec']}"
            )
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recalled = set(recall_rel_paths(output_dir, manifest["points"]))
    kept = [rel for rel in manifest["points"] if rel not in recalled]
    for rel in recalled:
        shutil.rmtree(output_dir / Path(rel).parent)
    new_paths = write_recall_points(
        source_id=output_dir.name,
        labels=labels,
        speaker_paths=speaker_paths,
        points_dir=output_dir / "points",
        personas_payload=load_personas(personas_path),
        channel_words=channel_words,
        start_index=len(kept),
    )
    manifest["points"] = kept + new_paths
    manifest["point_count"] = len(manifest["points"])
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return len(manifest["points"])


def process_datapoint(
    datapoint_dir: Path,
    output_root: Path,
    source_labels: str | None = None,
    source_audio: dict[str, str] | None = None,
    channel_words: dict[str, list[dict]] | None = None,
    max_points: int | None = None,
    recall_only: bool = False,
    chunk: dict | None = None,
    curation: dict[int, dict] | None = None,
) -> int:
    source_id = datapoint_dir.name
    output_dir = output_root / source_id
    if recall_only and (output_dir / "manifest.json").is_file():
        return write_recall_onto_existing(output_dir, channel_words, chunk)

    labels_path = datapoint_dir / "label.json"

    labels = load_labels(labels_path)
    split_events = iter_split_events(labels)
    speaker_paths = {
        speaker: speaker_audio_path(datapoint_dir, speaker)
        for speaker in ("speaker_1", "speaker_2")
    }
    for path in speaker_paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"{source_id}: missing {path}")

    output_dir = output_root / source_id
    if output_dir.exists():
        shutil.rmtree(output_dir)
    points_dir = output_dir / "points"
    points_dir.mkdir(parents=True)
    shutil.copy2(labels_path, output_dir / "label.json")
    waves_out = output_dir / "waves"
    waves_out.mkdir()
    for path in speaker_paths.values():
        shutil.copy2(path, waves_out / path.name)
    write_conversation_mix(output_dir)
    if chunk is None:
        duration_sec = min(wav_duration_sec(path) for path in speaker_paths.values())
        chunk = {
            "session_id": source_id,
            "chunk_index": 0,
            "chunk_count": 1,
            "source_start_sec": 0.0,
            "source_end_sec": duration_sec,
            "history_budget_sec": HISTORY_BUDGET_SEC,
        }
    with (output_dir / "chunk.json").open("w", encoding="utf-8") as handle:
        json.dump(chunk, handle, indent=2)
        handle.write("\n")

    if not channel_words:
        print(
            f"{source_id}: no transcripts, writing identity frame only",
            flush=True,
        )
        personas_payload = frame_only_personas()
    else:
        print(
            f"{source_id}: generating speaker personas",
            flush=True,
        )
        personas_payload = generate_personas(labels, channel_words)
    write_personas(output_dir / "personas.json", personas_payload)

    point_paths: list[str] = []
    point_index = 0
    if not recall_only:
        for event in split_events:
            if max_points is not None and point_index >= max_points:
                break
            reason = skip_reason(event, labels, speaker_paths)
            if reason is not None:
                print(
                    f"{source_id}: skipping split @ {event.timestamp:.3f}s "
                    f"({event.split_type}, {event.active_speaker}): {reason}",
                    flush=True,
                )
                continue
            decision = curation.get(event.index) if curation is not None else None
            if decision is not None and not decision["keep"]:
                print(
                    f"{source_id}: curation cut split @ {event.timestamp:.3f}s "
                    f"({event.split_type}, {event.active_speaker}): "
                    f"{decision['reason']}",
                    flush=True,
                )
                continue
            point_index += 1
            point_dir = points_dir / f"{point_index:06d}"
            point_dir.mkdir()
            point_paths.append(
                write_split_point(
                    point_dir,
                    source_id,
                    event,
                    labels,
                    speaker_paths,
                    personas_payload,
                    channel_words,
                    point_index,
                    curation_decision=decision,
                )
            )

    if max_points is None or recall_only:
        point_paths.extend(
            write_recall_points(
                source_id=source_id,
                labels=labels,
                speaker_paths=speaker_paths,
                points_dir=points_dir,
                personas_payload=personas_payload,
                channel_words=channel_words,
                start_index=point_index,
            )
        )

    manifest = {
        "source_id": source_id,
        "source_audio": source_audio
        or {
            speaker: rel_path(output_dir, path)
            for speaker, path in speaker_paths.items()
        },
        "source_labels": source_labels or rel_path(output_dir, labels_path),
        "point_count": len(point_paths),
        "points": point_paths,
        "personas": "personas.json",
        "label": "label.json",
        "waves": {
            "speaker_1": "waves/speaker_1.wav",
            "speaker_2": "waves/speaker_2.wav",
        },
        "conversation": "conversation.wav",
        "chunk": "chunk.json",
        "session_id": chunk["session_id"],
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    update_root_manifest(output_root, source_id, point_paths)

    return len(point_paths)


def update_root_manifest(
    output_root: Path, source_id: str, point_paths: list[str]
) -> None:
    """Maintain one cross-source manifest at the output root for bench eval.

    A fresh output gets a new manifest; when the output already holds one from
    earlier runs, only this source's entries are replaced, so other sources'
    points survive re-runs of a subset.
    """
    manifest_path = output_root / "manifest.json"
    points: list[str] = []
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            points = [
                point
                for point in json.load(handle)["points"]
                if not point.startswith(f"{source_id}/")
            ]
    points.extend(f"{source_id}/{path}" for path in point_paths)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump({"point_count": len(points), "points": points}, handle, indent=2)
        handle.write("\n")


def build_point_meta(
    point_id: str,
    source_id: str,
    event: SplitEvent,
    window: RolloutWindow,
    sample_rate: int,
    input_audio: str,
    text_prompt: str,
    include_context: bool = False,
    reference: ReferenceResponse | None = None,
    history_start: float = 0.0,
) -> dict:
    meta = {
        "id": point_id,
        "source_id": source_id,
        "split_timestamp": event.timestamp,
        "split_type": event.split_type,
        "user_speaker": event.active_speaker,
        "assistant_speaker": event.model_speaker,
        "context_duration_sec": window.input_end,
        "user_audio": "user.wav",
        "assistant_audio": "assistant.wav",
        "assistant_transcript": "assistant_transcript.json",
        "sample_rate": sample_rate,
        "channels": 1,
        "text_prompt": text_prompt,
    }
    if include_context:
        meta["context"] = "context.json"
    expected_action = "eot" if event.split_type == "end_of_turn" else "pause"
    meta["rollout"] = {
        "expected_action": expected_action,
        "history_start_sec": history_start,
        "history_end_sec": window.turn_start,
        "input_end_sec": window.input_end,
        "checkpoint_sec": window.checkpoint - window.turn_start,
        "window_end_sec": window.window_end - window.turn_start,
        "user_audio": "rollout_user_context.wav",
        "assistant_audio": "rollout_assistant.wav",
        "assistant_transcript": "rollout_assistant_transcript.json",
        "input_audio": input_audio,
    }
    if include_context:
        meta["rollout"]["context"] = "rollout_context.json"
    if expected_action == "eot":
        if reference is None:
            raise ValueError("eot points require a reference response")
        meta["rollout"]["reference_audio"] = reference.audio
        meta["rollout"]["reference_transcript"] = reference.transcript
        meta["rollout"]["reference_start_sec"] = reference.start_sec
        meta["rollout"]["reference_end_sec"] = reference.end_sec
        meta["rollout"]["reference_duration_sec"] = reference.duration_sec
        meta["rollout"]["reference_word_count"] = reference.word_count
    return meta


def rel_path(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target, from_dir)).as_posix()


def merge_session_index(path: Path, rows: list[dict]) -> None:
    existing: list[dict] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8")).get("datapoints", [])
    replaced = {row["session_id"] for row in rows}
    merged = [row for row in existing if row.get("session_id") not in replaced]
    merged.extend(rows)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"datapoints": merged}, handle, indent=2)
        handle.write("\n")


def run(
    output_dir: str | Path,
    datapoints: list[str] | None = None,
    profile: str = AWS_PROFILE,
    datasource: str = DEFAULT_DATASOURCE,
    limit: int | None = None,
    max_points: int | None = None,
    max_datapoints: int | None = None,
    recall_only: bool = False,
    curation_dir: str | Path | None = None,
) -> None:
    output_path = Path(output_dir)
    source = get_source(datasource)
    s3 = boto3.Session(profile_name=profile).client("s3")
    available = source.list_names(s3)
    if datapoints:
        source_names = [resolve_datapoint(name, available) for name in datapoints]
    else:
        source_names = available
    if limit is not None:
        source_names = source_names[:limit]
    if not source_names:
        print(f"No datapoints found for datasource {source.name}")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    total_points = 0
    datapoint_count = 0
    written: list[dict] = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        for source_name in source_names:
            if max_datapoints is not None and datapoint_count >= max_datapoints:
                break
            session_dir = temp_root / source_name.replace("-", "_")
            loaded = source.load(s3, source_name, session_dir)
            curation = (
                load_curation(Path(curation_dir), loaded.source_id)
                if curation_dir
                else None
            )
            chunk_temp = temp_root / f"{loaded.source_id}_chunks"
            chunk_temp.mkdir()
            chunks = iter_session_chunks(
                session_dir,
                loaded.source_id,
                loaded.channel_words,
                chunk_temp,
            )
            print(
                f"{loaded.source_id}: {len(chunks)} chunk datapoint(s)",
                flush=True,
            )
            for chunk_dir, words, chunk in chunks:
                if max_datapoints is not None and datapoint_count >= max_datapoints:
                    break
                point_count = process_datapoint(
                    chunk_dir,
                    output_path,
                    source_labels=loaded.source_labels,
                    source_audio=loaded.source_audio,
                    channel_words=words,
                    max_points=max_points,
                    recall_only=recall_only,
                    chunk=chunk,
                    curation=curation,
                )
                datapoint_count += 1
                total_points += point_count
                dest = output_path / chunk_dir.name
                written.append(
                    {
                        "source_id": chunk_dir.name,
                        "session_id": chunk["session_id"],
                        "path": dest.as_posix(),
                        "source_start_sec": chunk["source_start_sec"],
                        "source_end_sec": chunk["source_end_sec"],
                        "point_count": point_count,
                    }
                )
                print(
                    f"{chunk_dir.name}: {chunk['source_start_sec']:.1f}s-"
                    f"{chunk['source_end_sec']:.1f}s, {point_count} eval "
                    f"points -> {dest}",
                    flush=True,
                )

    merge_session_index(output_path / "sessions.json", written)
    print(
        f"Done. {total_points} eval points across {datapoint_count} "
        f"datapoint(s) from {len(source_names)} session(s).",
    )
