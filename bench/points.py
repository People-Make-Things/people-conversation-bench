"""Load role-separated eval points for context preloading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from bench.audio import (
    MODEL_FRAME_SAMPLES,
    TARGET_SAMPLE_RATE,
    pad_pcm_to_model_frames,
    read_wav_segment,
)
from bench.protocol import (
    Context,
    ContextType,
    DuplexAudioContext,
    EventContext,
    PreloadNotSupportedError,
    TimedWord,
    event_context,
)
from utils.timed_words import load_timed_words, words_text

HOLD_DRAIN = "hold"
RESPOND_DRAIN = "respond"

PAUSE_EVAL = "pause-recognition"
LENGTH_EVAL = "response-length"
SIMILARITY_EVAL = "response-similarity"
LATENCY_EVAL = "response-latency"
CONVERSATION_RECALL_EVAL = "conversation-recall"
FACT_RECALL_EVAL = "fact-recall"


@dataclass(frozen=True)
class ScenarioSpec:
    eval_type: str
    metrics: tuple[str, ...]
    drain: str
    detect_turns: bool


SPECS: dict[str, ScenarioSpec] = {
    "pause": ScenarioSpec(PAUSE_EVAL, (PAUSE_EVAL,), HOLD_DRAIN, True),
    "eot": ScenarioSpec(
        LENGTH_EVAL, (LENGTH_EVAL, SIMILARITY_EVAL), RESPOND_DRAIN, False
    ),
    "conversation_recall": ScenarioSpec(
        CONVERSATION_RECALL_EVAL, (CONVERSATION_RECALL_EVAL,), RESPOND_DRAIN, False
    ),
    "fact_recall": ScenarioSpec(
        FACT_RECALL_EVAL, (FACT_RECALL_EVAL,), RESPOND_DRAIN, False
    ),
}

ACTION_EVAL_TYPES = {action: spec.eval_type for action, spec in SPECS.items()}
EVAL_TYPES = tuple(dict.fromkeys(spec.eval_type for spec in SPECS.values()))
RECALL_EVAL_TYPES = (CONVERSATION_RECALL_EVAL, FACT_RECALL_EVAL)
RECALL_ACTIONS = frozenset({"conversation_recall", "fact_recall"})
REPORT_SECTIONS = (*EVAL_TYPES, SIMILARITY_EVAL)
SPLIT_TYPE_ACTIONS = {
    "pause_start": "pause",
    "end_of_turn": "eot",
}
DEFAULT_CONTEXT_KINDS = (
    ContextType.DUPLEX_AUDIO,
    ContextType.TEXT,
    ContextType.TEXT_AUDIO,
)
INTERACT_FILES = {
    "user_audio": "user.wav",
    "assistant_audio": "assistant.wav",
    "assistant_transcript": "assistant_transcript.json",
}
ROLLOUT_FILES = {
    "user_audio": "rollout_user_context.wav",
    "assistant_audio": "rollout_assistant.wav",
    "assistant_transcript": "rollout_assistant_transcript.json",
}


@dataclass(frozen=True)
class ContextFiles:
    user_audio: str
    assistant_audio: str
    assistant_transcript: str
    events: str | None = None


@dataclass(frozen=True)
class ReferenceResponse:
    duration_sec: float
    word_count: int
    text: str = ""
    audio: str = ""
    transcript: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0


@dataclass(frozen=True)
class PointMeta:
    point_id: str
    sample_rate: int
    text_prompt: str | None
    interact: ContextFiles
    rollout: dict | None


@dataclass(frozen=True)
class EvalPoint:
    point_id: str
    point_dir: Path
    meta: dict
    contexts: dict[ContextType, Context]
    text_prompt: str | None = None


@dataclass(frozen=True, kw_only=True)
class LiveTurn:
    point_id: str
    point_dir: Path
    contexts: dict[ContextType, Context]
    input_pcm: np.ndarray
    sample_rate: int
    checkpoint_sec: float
    window_end_sec: float
    text_prompt: str | None = None

    @property
    def spec(self) -> ScenarioSpec:
        return SPECS[self.expected_action]

    @property
    def eval_type(self) -> str:
        return self.spec.eval_type

    @property
    def metrics(self) -> tuple[str, ...]:
        return self.spec.metrics

    @property
    def drain(self) -> str:
        return self.spec.drain

    @property
    def detect_turns(self) -> bool:
        return self.spec.detect_turns

    def live_pcm(self, *, step_driven: bool) -> np.ndarray:
        return self.input_pcm


@dataclass(frozen=True, kw_only=True)
class PauseScenario(LiveTurn):
    """Live turn through pause_end, plus the labeled hold window."""

    expected_action: ClassVar[str] = "pause"
    pause_input_pcm: np.ndarray

    def live_pcm(self, *, step_driven: bool) -> np.ndarray:
        # Both families stream the real pause audio through the hold window: a
        # step-driven model needs the LM steps, and a hosted model's server VAD
        # only closes the user turn on silence it actually receives.
        if step_driven:
            return pad_pcm_to_model_frames(self.pause_input_pcm, MODEL_FRAME_SAMPLES)
        return self.pause_input_pcm


@dataclass(frozen=True, kw_only=True)
class EotScenario(LiveTurn):
    """Live user turn through turn end, plus the human assistant's next turn."""

    expected_action: ClassVar[str] = "eot"
    reference: ReferenceResponse


@dataclass(frozen=True, kw_only=True)
class RecallScenario(LiveTurn):
    """Synthetic probe after a conversation prefix, scored for factual recall."""

    expected_action: str
    question: str
    history_target_sec: float
    history_sec: float
    gold_answer: str = ""
    conversation: str = ""


RolloutScenario = PauseScenario | EotScenario | RecallScenario


def normalize_expected_action(action: str) -> str:
    if action not in SPECS:
        raise ValueError(f"unknown expected action: {action}")
    return action


def eval_type_for_action(action: str) -> str:
    return SPECS[normalize_expected_action(action)].eval_type


def eval_type_for_point(point_meta: dict) -> str | None:
    rollout = point_meta.get("rollout")
    action = rollout.get("expected_action") if isinstance(rollout, dict) else None
    if not action:
        split_type = point_meta.get("split_type")
        action = (
            SPLIT_TYPE_ACTIONS.get(split_type) if isinstance(split_type, str) else None
        )
    if not action:
        return None
    try:
        return eval_type_for_action(str(action))
    except ValueError:
        return None


def validate_history(
    user_pcm: np.ndarray,
    assistant_pcm: np.ndarray,
    sample_rate: int,
    words: tuple[TimedWord, ...],
) -> None:
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"expected sample rate {TARGET_SAMPLE_RATE}, got {sample_rate}")
    if user_pcm.ndim != 1 or assistant_pcm.ndim != 1:
        raise ValueError("history audio must be mono 1-D arrays")
    if user_pcm.shape[0] != assistant_pcm.shape[0]:
        raise ValueError(
            "user and assistant history must have equal sample counts: "
            f"{user_pcm.shape[0]} vs {assistant_pcm.shape[0]}"
        )
    if user_pcm.shape[0] % MODEL_FRAME_SAMPLES != 0:
        raise ValueError(
            f"history audio must be a multiple of {MODEL_FRAME_SAMPLES} samples"
        )
    duration_sec = user_pcm.shape[0] / sample_rate
    for word in words:
        if word.start_sec < -1e-6 or word.start_sec > duration_sec + 1e-6:
            raise ValueError(
                f"word '{word.text}' start_sec {word.start_sec} outside history"
            )
        if word.end_sec is not None and word.end_sec > duration_sec + 1e-6:
            raise ValueError(
                f"word '{word.text}' end_sec {word.end_sec} outside history"
            )


def load_event_context(point_dir: Path, filename: str) -> EventContext | None:
    context_path = point_dir / filename
    if not context_path.is_file():
        return None
    with context_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    events = payload.get("events", payload)
    if not isinstance(events, list):
        raise ValueError(f"{context_path}: expected a list of events")
    return event_context(events)


def conversation_text(events: tuple[dict, ...] | list[dict]) -> str:
    lines: list[str] = []
    for event in events:
        item = event["item"]
        role = str(item["role"])
        for part in item["content"]:
            text = str(part.get("text", "")).strip()
            if text:
                lines.append(f"{role}: {text}")
    return "\n".join(lines)


def needs_duplex_context(kinds: tuple[ContextType, ...]) -> bool:
    return ContextType.DUPLEX_AUDIO in kinds


def needs_event_context(kinds: tuple[ContextType, ...]) -> bool:
    return ContextType.TEXT in kinds or ContextType.TEXT_AUDIO in kinds


def select_context(
    contexts: dict[ContextType, Context],
    supported: tuple[ContextType, ...],
    point_id: str = "",
) -> Context:
    for kind in supported:
        if kind in contexts:
            return contexts[kind]
    available = [kind.value for kind in contexts]
    supported_names = [kind.value for kind in supported]
    label = f"point {point_id} " if point_id else ""
    raise PreloadNotSupportedError(
        f"Model supports {supported_names} but {label}only provides {available}"
    )


def point_text_prompt(meta: dict) -> str | None:
    prompt = meta.get("text_prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    return prompt


def read_point_file(point_path: str | Path) -> tuple[Path, Path, dict]:
    point_json = Path(point_path)
    if point_json.is_dir():
        point_json = point_json / "point.json"
    if not point_json.is_file():
        raise FileNotFoundError(f"point metadata not found: {point_json}")
    with point_json.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    return point_json, point_json.parent, meta


def parse_context_files(
    data: dict,
    defaults: dict[str, str],
    transcript_fallback: str | None = None,
) -> ContextFiles:
    events = data.get("context")
    transcript = data.get("assistant_transcript", transcript_fallback)
    if transcript is None:
        transcript = defaults["assistant_transcript"]
    return ContextFiles(
        user_audio=str(data.get("user_audio", defaults["user_audio"])),
        assistant_audio=str(data.get("assistant_audio", defaults["assistant_audio"])),
        assistant_transcript=str(transcript),
        events=str(events) if events else None,
    )


def parse_point_meta(meta: dict, *, default_id: str) -> PointMeta:
    return PointMeta(
        point_id=str(meta.get("id", default_id)),
        sample_rate=int(meta.get("sample_rate", TARGET_SAMPLE_RATE)),
        text_prompt=point_text_prompt(meta),
        interact=parse_context_files(meta, INTERACT_FILES),
        rollout=meta.get("rollout") if isinstance(meta.get("rollout"), dict) else None,
    )


def load_contexts(
    point_dir: Path,
    files: ContextFiles,
    kinds: tuple[ContextType, ...],
    sample_rate: int,
) -> dict[ContextType, Context]:
    contexts: dict[ContextType, Context] = {}
    if files.events and needs_event_context(kinds):
        event_ctx = load_event_context(point_dir, files.events)
        if event_ctx is not None and event_ctx.kind in kinds:
            contexts[event_ctx.kind] = event_ctx
    if needs_duplex_context(kinds):
        user_pcm, _ = read_wav_segment(
            point_dir / files.user_audio, target_rate=sample_rate
        )
        assistant_pcm, _ = read_wav_segment(
            point_dir / files.assistant_audio, target_rate=sample_rate
        )
        words = load_timed_words(point_dir / files.assistant_transcript)
        validate_history(user_pcm, assistant_pcm, sample_rate, words)
        contexts[ContextType.DUPLEX_AUDIO] = DuplexAudioContext(
            user_pcm=user_pcm,
            assistant_pcm=assistant_pcm,
            sample_rate=sample_rate,
            words=words,
        )
    return contexts


def load_point(
    point_path: str | Path,
    kinds: tuple[ContextType, ...] | None = None,
) -> EvalPoint:
    _point_json, point_dir, meta = read_point_file(point_path)
    parsed = parse_point_meta(meta, default_id=point_dir.name)
    requested = kinds or DEFAULT_CONTEXT_KINDS
    return EvalPoint(
        point_id=parsed.point_id,
        point_dir=point_dir,
        meta=meta,
        contexts=load_contexts(
            point_dir, parsed.interact, requested, parsed.sample_rate
        ),
        text_prompt=parsed.text_prompt,
    )


def load_rollout_scenario(
    point_path: str | Path,
    kinds: tuple[ContextType, ...] | None = None,
    meta: dict | None = None,
) -> RolloutScenario:
    if meta is None:
        point_json, point_dir, meta = read_point_file(point_path)
    else:
        point_json = Path(point_path)
        if point_json.is_dir():
            point_json = point_json / "point.json"
        point_dir = point_json.parent
    parsed = parse_point_meta(meta, default_id=point_dir.name)
    rollout = parsed.rollout
    if rollout is None:
        raise ValueError(f"{point_json}: missing rollout metadata")

    sample_rate = parsed.sample_rate
    requested = kinds or DEFAULT_CONTEXT_KINDS
    files = parse_context_files(
        rollout,
        ROLLOUT_FILES,
        transcript_fallback=meta.get(
            "assistant_transcript", ROLLOUT_FILES["assistant_transcript"]
        ),
    )
    contexts = load_contexts(point_dir, files, requested, sample_rate)
    input_pcm, _ = read_wav_segment(
        point_dir / rollout["input_audio"],
        target_rate=sample_rate,
    )
    expected_action = normalize_expected_action(str(rollout["expected_action"]))
    checkpoint_sec = float(rollout["checkpoint_sec"])
    window_end_sec = float(rollout["window_end_sec"])
    duration_sec = input_pcm.shape[0] / sample_rate
    tolerance = 1 / sample_rate
    frame_sec = MODEL_FRAME_SAMPLES / sample_rate
    if checkpoint_sec < 0 or checkpoint_sec > duration_sec + tolerance:
        raise ValueError(f"{point_json}: checkpoint is outside rollout input")
    if duration_sec - checkpoint_sec > frame_sec + tolerance:
        raise ValueError(
            f"{point_json}: rollout input runs {duration_sec - checkpoint_sec:.3f}s "
            "past the checkpoint; it should end there up to one frame of padding"
        )
    if window_end_sec < checkpoint_sec:
        raise ValueError(f"{point_json}: evaluation window ends before checkpoint")

    shared = {
        "point_id": parsed.point_id,
        "point_dir": point_dir,
        "contexts": contexts,
        "input_pcm": input_pcm,
        "sample_rate": sample_rate,
        "checkpoint_sec": checkpoint_sec,
        "window_end_sec": window_end_sec,
        "text_prompt": parsed.text_prompt,
    }
    if expected_action in RECALL_ACTIONS:
        question = str(rollout.get("question", "")).strip()
        if not question:
            raise ValueError(f"{point_json}: recall point missing question")
        gold_answer = str(rollout.get("gold_answer", "")).strip()
        conversation = ""
        if expected_action == "conversation_recall":
            filename = str(rollout.get("context", "rollout_context.json"))
            context = load_event_context(point_dir, filename)
            if context is None:
                raise ValueError(
                    f"{point_json}: conversation-recall missing {filename}"
                )
            conversation = conversation_text(context.events)
            if not conversation:
                raise ValueError(
                    f"{point_json}: conversation-recall context is empty"
                )
        elif not gold_answer:
            raise ValueError(f"{point_json}: fact-recall point missing gold_answer")
        return RecallScenario(
            **shared,
            expected_action=expected_action,
            question=question,
            gold_answer=gold_answer,
            history_target_sec=float(rollout["history_target_sec"]),
            history_sec=float(rollout["history_sec"]),
            conversation=conversation,
        )
    if expected_action == "eot":
        if "reference_duration_sec" not in rollout or "reference_word_count" not in rollout:
            raise ValueError(
                f"{point_json}: eot point missing reference response; re-run prepare-eval"
            )
        transcript_name = rollout.get(
            "reference_transcript", "rollout_reference_response_transcript.json"
        )
        transcript_path = point_dir / transcript_name
        if not transcript_path.is_file():
            raise ValueError(
                f"{point_json}: eot point missing reference transcript; "
                "re-run prepare-eval"
            )
        return EotScenario(
            **shared,
            reference=ReferenceResponse(
                duration_sec=float(rollout["reference_duration_sec"]),
                word_count=int(rollout["reference_word_count"]),
                text=words_text(load_timed_words(transcript_path)),
                audio=str(rollout.get("reference_audio", "")),
                transcript=transcript_name,
                start_sec=float(rollout.get("reference_start_sec", 0.0)),
                end_sec=float(rollout.get("reference_end_sec", 0.0)),
            ),
        )

    pause_input_path = point_dir / rollout.get(
        "input_audio_pause_end", "rollout_user_input_pause_end.wav"
    )
    if not pause_input_path.is_file():
        raise ValueError(
            f"{point_json}: pause point missing {pause_input_path.name}"
        )
    pause_input_pcm, _ = read_wav_segment(pause_input_path, target_rate=sample_rate)
    pause_duration_sec = pause_input_pcm.shape[0] / sample_rate
    if window_end_sec > pause_duration_sec + tolerance:
        raise ValueError(
            f"{point_json}: evaluation window ends after {pause_input_path.name}"
        )
    return PauseScenario(**shared, pause_input_pcm=pause_input_pcm)
