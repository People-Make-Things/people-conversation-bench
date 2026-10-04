"""Prepare conversation-recall and fact-recall eval points."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path

import numpy as np

from bench.audio import TARGET_SAMPLE_RATE, write_wav_pcm16
from bench.protocol import TimedWord
from data_processing.audio import read_wav_range, wav_duration_sec
from data_processing.cuts import HISTORY_TARGETS_SEC, recall_history_cuts
from data_processing.point_audio import transcript_for, write_duplex_history
from data_processing.context import (
    build_prefix_context_events,
    extract_turn_text,
    write_context,
)
from data_processing.labels import Turn, iter_prefix_turns, iter_turns
from data_processing.personas import DEFAULT_PERSONA_MODEL, persona_model_name
from data_processing.tts import synthesize_clone
from utils.openai import openai_json_complete

HISTORY_FILES = (
    "rollout_user_context.wav",
    "rollout_assistant.wav",
    "rollout_assistant_transcript.json",
)

RECALL_USER = "speaker_2"
RECALL_ASSISTANT = "speaker_1"
CONVERSATION_FACT_WINDOW_SEC = HISTORY_TARGETS_SEC[0]
MIN_REFERENCE_TURN_SEC = 1.0
MIN_REFERENCE_SEC = 20.0
MAX_REFERENCE_SEC = 40.0

FACT_BANK = (
    ("Who was the first president of the United States?", "George Washington"),
    ("What planet is closest to the sun?", "Mercury"),
    ("How many days are in a week?", "seven"),
    ("What is the capital of France?", "Paris"),
    ("How many legs does a spider have?", "eight"),
    ("What do you call frozen water?", "ice"),
    ("What color do you get by mixing blue and yellow?", "green"),
    ("What is the largest ocean on Earth?", "the Pacific"),
    ("How many minutes are in an hour?", "sixty"),
    ("What animal is known as man's best friend?", "the dog"),
    ("Which month comes right after April?", "May"),
    ("How many sides does a triangle have?", "three"),
)

FACT_EXTRACT_SYSTEM = """\
You extract one factual question from a conversation transcript.

Return JSON with keys question and answer. Both are plain strings.

The question must be answerable only from the transcript. The answer is the
short fact, not a recap of the whole conversation.

The question will be spoken aloud by speaker_2 to speaker_1. Phrase it from
speaker_2's perspective: say "I" or "me" for speaker_2 and "you" for
speaker_1. Never use the literal labels speaker_1 or speaker_2.

Do not invent facts. If nothing specific was said, return {"question": "", "answer": ""}."""

FACT_JUDGE_SYSTEM = """\
You judge whether an extracted conversation fact is a fair memory probe.

Return JSON with keys keep (0 or 1) and reason (a short string).

Keep only facts that are specific and memorable: a concrete detail unique to
this conversation (a name, a place, a number, an event, a stated preference)
that an attentive listener would retain. Reject vague or generic facts
("they introduced themselves", "they discussed hobbies"), facts about the
recording setup or the task itself, and questions whose answer is not
actually stated in the transcript."""

STYLE_SYSTEM = """\
You rewrite a question as one spoken turn in this speaker's style.

Return JSON with key text. Use their fillers and formality. One or two
sentences, spoken aloud, not narrated. Do not change what is being asked.
Do not add a recap or the answer."""

Synthesize = Callable[[np.ndarray, int, str, str], np.ndarray]
Complete = Callable[[str, str], str]
Transcribe = Callable[[Path], list[TimedWord]]


def format_prefix_transcript(
    labels: dict,
    channel_words: dict[str, list[dict]],
    end_sec: float,
) -> str:
    lines: list[str] = []
    for turn in iter_prefix_turns(labels, end_sec):
        text = extract_turn_text(
            channel_words.get(turn.speaker, []),
            turn.start,
            turn.end,
        ).strip()
        if text:
            lines.append(f"{turn.speaker}: {text}")
    return "\n".join(lines)


def fact_for_source(source_id: str) -> tuple[str, str]:
    digest = hashlib.sha256(source_id.encode("utf-8")).hexdigest()
    index = int(digest, 16) % len(FACT_BANK)
    return FACT_BANK[index]


def parse_question_answer(text: str) -> tuple[str, str] | None:
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("recall response must be a JSON object")
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    if not question or not answer:
        return None
    return question, answer


def extract_conversation_fact(
    transcript: str,
    *,
    complete: Complete,
) -> tuple[str, str] | None:
    if not transcript.strip():
        return None
    return parse_question_answer(
        complete(FACT_EXTRACT_SYSTEM, f"Transcript:\n{transcript}")
    )


def judge_conversation_fact(
    transcript: str,
    question: str,
    answer: str,
    *,
    complete: Complete,
) -> tuple[bool, str]:
    payload = json.loads(
        complete(
            FACT_JUDGE_SYSTEM,
            f"Transcript:\n{transcript}\n\nQuestion: {question}\nAnswer: {answer}",
        )
    )
    if not isinstance(payload, dict):
        raise ValueError("fact judge response must be a JSON object")
    return bool(int(payload.get("keep", 0))), str(payload.get("reason", ""))


def style_probe(
    question: str,
    speaker_turns: str,
    *,
    complete: Complete,
) -> str:
    prompt = f"Speaker's recent turns:\n{speaker_turns or '(none)'}\n\nQuestion:\n{question}"
    payload = json.loads(complete(STYLE_SYSTEM, prompt))
    if not isinstance(payload, dict):
        raise ValueError("style rewrite must be a JSON object")
    text = str(payload.get("text", "")).strip()
    return text or question


def speaker_turn_text(
    labels: dict,
    channel_words: dict[str, list[dict]] | None,
    speaker: str,
    end_sec: float | None = None,
) -> str:
    lines: list[str] = []
    words = (channel_words or {}).get(speaker, [])
    turns = (
        iter_prefix_turns(labels, end_sec)
        if end_sec is not None
        else iter_turns(labels, speaker)
    )
    for turn in turns:
        if turn.speaker != speaker:
            continue
        text = extract_turn_text(words, turn.start, turn.end).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def reference_turns(
    labels: dict,
    speaker: str,
    *,
    min_turn_sec: float = MIN_REFERENCE_TURN_SEC,
    min_sec: float = MIN_REFERENCE_SEC,
    max_sec: float = MAX_REFERENCE_SEC,
) -> list[Turn] | None:
    selected: list[Turn] = []
    total = 0.0
    for turn in iter_turns(labels, speaker):
        duration = turn.end - turn.start
        if duration < min_turn_sec:
            continue
        selected.append(turn)
        total += duration
        if total >= max_sec:
            break
    if total < min_sec:
        return None
    return selected


def load_reference_clip(
    speaker_path: Path,
    turns: list[Turn],
    channel_words: dict[str, list[dict]] | None,
    speaker: str,
) -> tuple[np.ndarray, int, str]:
    chunks: list[np.ndarray] = []
    texts: list[str] = []
    sample_rate = TARGET_SAMPLE_RATE
    words = (channel_words or {}).get(speaker, [])
    for turn in turns:
        pcm, sample_rate = read_wav_range(speaker_path, turn.start, turn.end)
        if pcm.size:
            chunks.append(pcm)
        text = extract_turn_text(words, turn.start, turn.end).strip()
        if text:
            texts.append(text)
    if not chunks:
        raise ValueError("reference clip has no audio")
    return np.concatenate(chunks), sample_rate, " ".join(texts)


def build_recall_point_meta(
    point_id: str,
    source_id: str,
    *,
    expected_action: str,
    history_target_sec: float,
    history_sec: float,
    checkpoint_sec: float,
    sample_rate: int,
    text_prompt: str,
    question: str,
    gold_answer: str,
    include_context: bool,
) -> dict:
    meta = {
        "id": point_id,
        "source_id": source_id,
        "split_timestamp": history_sec,
        "split_type": expected_action,
        "user_speaker": RECALL_USER,
        "assistant_speaker": RECALL_ASSISTANT,
        "context_duration_sec": history_sec,
        "sample_rate": sample_rate,
        "channels": 1,
        "text_prompt": text_prompt,
        "rollout": {
            "expected_action": expected_action,
            "history_end_sec": history_sec,
            "input_end_sec": history_sec,
            "checkpoint_sec": checkpoint_sec,
            "window_end_sec": checkpoint_sec,
            "history_target_sec": history_target_sec,
            "history_sec": history_sec,
            "question": question,
            "user_audio": "rollout_user_context.wav",
            "assistant_audio": "rollout_assistant.wav",
            "assistant_transcript": "rollout_assistant_transcript.json",
            "input_audio": "rollout_user_input.wav",
        },
    }
    if expected_action == "fact_recall":
        meta["rollout"]["gold_answer"] = gold_answer
    if include_context:
        meta["rollout"]["context"] = "rollout_context.json"
    return meta


def copy_duplex_history(source_dir: Path, dest_dir: Path) -> None:
    for name in HISTORY_FILES:
        shutil.copy2(source_dir / name, dest_dir / name)


def styled_recall_probes(
    source_id: str,
    labels: dict,
    channel_words: dict[str, list[dict]] | None,
    speaker: str,
    complete: Complete,
    fact_window_sec: float | None = None,
) -> list[tuple[str, str, str]]:
    fact_question, fact_answer = fact_for_source(source_id)
    probes = [
        (
            "fact_recall",
            style_probe(
                fact_question,
                speaker_turn_text(labels, channel_words, speaker),
                complete=complete,
            ),
            fact_answer,
        )
    ]
    if not channel_words or fact_window_sec is None:
        return probes
    transcript = format_prefix_transcript(labels, channel_words, fact_window_sec)
    conversation = extract_conversation_fact(transcript, complete=complete)
    if conversation is None:
        return probes
    question, answer = conversation
    keep, reason = judge_conversation_fact(
        transcript, question, answer, complete=complete
    )
    if not keep:
        print(
            f"{source_id}: conversation fact rejected by judge: {reason}",
            flush=True,
        )
        return probes
    spoken = style_probe(
        question,
        speaker_turn_text(
            labels, channel_words, speaker, fact_window_sec
        ),
        complete=complete,
    )
    if "speaker_" in spoken.lower():
        print(
            f"{source_id}: conversation probe leaks a speaker label, "
            f"skipping: {spoken!r}",
            flush=True,
        )
        return probes
    probes.append(("conversation_recall", spoken, answer))
    return probes


def write_recall_points(
    *,
    source_id: str,
    labels: dict,
    speaker_paths: dict[str, Path],
    points_dir: Path,
    personas_payload: dict,
    channel_words: dict[str, list[dict]] | None,
    start_index: int = 0,
    synthesize: Synthesize = synthesize_clone,
    complete: Complete | None = None,
    transcribe: Transcribe | None = None,
) -> list[str]:
    existing = [path for path in speaker_paths.values() if path.is_file()]
    duration_sec = (
        min(wav_duration_sec(path) for path in existing) if existing else None
    )
    cuts = recall_history_cuts(labels, duration_sec=duration_sec)
    if not cuts:
        print(f"{source_id}: no prefix long enough for recall points", flush=True)
        return []

    turns = reference_turns(labels, RECALL_USER)
    if turns is None:
        print(
            f"{source_id}: skipping recall, {RECALL_USER} has under "
            f"{MIN_REFERENCE_SEC:.0f}s of usable speech",
            flush=True,
        )
        return []

    reference_pcm, reference_rate, reference_text = load_reference_clip(
        speaker_paths[RECALL_USER],
        turns,
        channel_words,
        RECALL_USER,
    )
    model_name = persona_model_name()
    completer = complete or (
        lambda system, prompt: openai_json_complete(
            system, prompt, model=model_name or DEFAULT_PERSONA_MODEL
        )
    )
    fact_window_sec = next(
        (history_sec for target_sec, history_sec in cuts if target_sec == HISTORY_TARGETS_SEC[0]),
        None,
    )
    probes = styled_recall_probes(
        source_id,
        labels,
        channel_words,
        RECALL_USER,
        completer,
        fact_window_sec=fact_window_sec,
    )
    if channel_words and all(action != "conversation_recall" for action, *_ in probes):
        print(
            f"{source_id}: no conversation fact in the first "
            f"{CONVERSATION_FACT_WINDOW_SEC:.0f}s, skipping conversation-recall",
            flush=True,
        )

    # A missing TTS extra raises here (RuntimeError with the install hint):
    # an environment that cannot synthesize probes must fail the run loudly
    # rather than silently produce a dataset with no recall points.
    probe_pcm = {
        action: synthesize(reference_pcm, reference_rate, reference_text, spoken)
        for action, spoken, _answer in probes
    }

    text_prompt = personas_payload[RECALL_ASSISTANT]["text_prompt"]
    sample_rate = TARGET_SAMPLE_RATE
    include_context = channel_words is not None
    written_history: dict[float, Path] = {}
    point_paths: list[str] = []
    point_index = start_index

    for target_sec, history_sec in cuts:
        for action, spoken, gold_answer in probes:
            point_index += 1
            point_dir = points_dir / f"{point_index:06d}"
            point_dir.mkdir()
            cached = written_history.get(history_sec)
            if cached is None:
                write_duplex_history(
                    point_dir,
                    speaker_paths[RECALL_USER],
                    speaker_paths[RECALL_ASSISTANT],
                    history_sec,
                    sample_rate,
                    transcribe=transcript_for(
                        channel_words,
                        RECALL_ASSISTANT,
                        0.0,
                        history_sec,
                        transcribe=transcribe,
                    ),
                )
                written_history[history_sec] = point_dir
            else:
                copy_duplex_history(cached, point_dir)
            pcm = probe_pcm[action]
            write_wav_pcm16(point_dir / "rollout_user_input.wav", pcm, sample_rate)
            if include_context:
                write_context(
                    point_dir / "rollout_context.json",
                    build_prefix_context_events(
                        labels, channel_words, RECALL_USER, history_sec
                    ),
                )
            checkpoint_sec = pcm.shape[0] / sample_rate
            meta = build_recall_point_meta(
                f"{source_id}_{point_index:06d}",
                source_id,
                expected_action=action,
                history_target_sec=target_sec,
                history_sec=history_sec,
                checkpoint_sec=checkpoint_sec,
                sample_rate=sample_rate,
                text_prompt=text_prompt,
                question=spoken,
                gold_answer=gold_answer,
                include_context=include_context,
            )
            with (point_dir / "point.json").open("w", encoding="utf-8") as handle:
                json.dump(meta, handle, indent=2)
                handle.write("\n")
            point_paths.append(f"points/{point_index:06d}/point.json")

    return point_paths
