"""Conversation-recall and fact-recall scoring for rollout traces."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from bench.metrics.length import eot_onset
from bench.points import ACTION_EVAL_TYPES, RECALL_ACTIONS, RECALL_EVAL_TYPES
from bench.speech import post_checkpoint_intervals
from bench.trace import SCORED, RolloutTrace
from utils.openai import openai_json_complete

DEFAULT_JUDGE_MODEL = "gpt-4o"

JUDGE_SYSTEM = """\
You grade whether a spoken answer correctly recalls a fact.

Return JSON with key correct (boolean). The answer is correct if it contains
the gold fact or an unambiguous equivalent. Hedging is fine. A recap that
omits the fact is incorrect. An empty transcript is incorrect."""

CONVERSATION_JUDGE_SYSTEM = """\
You grade whether a spoken answer correctly recalls information from a
conversation.

Return JSON with key correct (boolean). Use only the conversation. The answer
is correct if it is supported by the conversation and addresses the question.
Hedging is fine. A recap that omits what was asked is incorrect. An empty
transcript is incorrect. World knowledge does not count."""


@dataclass(frozen=True)
class RecallScore:
    eval: str
    status: str
    score: int | None
    question: str
    gold_answer: str
    transcript: str
    history_target_sec: float
    history_sec: float
    timed_out: bool
    conversation: str = ""
    latency_ms: float | None = None
    wordless_phonation: bool = False

    def to_dict(self) -> dict:
        payload = {
            "eval": self.eval,
            "status": self.status,
            "score": self.score,
            "question": self.question,
            "transcript": self.transcript,
            "history_target_sec": self.history_target_sec,
            "history_sec": self.history_sec,
            "timed_out": self.timed_out,
            "latency_ms": self.latency_ms,
            "wordless_phonation": self.wordless_phonation,
        }
        if self.gold_answer:
            payload["gold_answer"] = self.gold_answer
        if self.conversation:
            payload["conversation"] = self.conversation
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> RecallScore:
        score = data.get("score")
        return cls(
            eval=str(data["eval"]),
            status=str(data.get("status", SCORED)),
            score=None if score is None else int(score),
            question=str(data.get("question", "")),
            gold_answer=str(data.get("gold_answer", "")),
            transcript=str(data.get("transcript", "")),
            history_target_sec=float(data["history_target_sec"]),
            history_sec=float(data["history_sec"]),
            timed_out=bool(data.get("timed_out", False)),
            conversation=str(data.get("conversation", "")),
            latency_ms=(
                None if data.get("latency_ms") is None else float(data["latency_ms"])
            ),
            wordless_phonation=bool(data.get("wordless_phonation", False)),
        )


def recall_judge_request(
    eval_name: str,
    *,
    question: str,
    transcript: str,
    gold_answer: str = "",
    conversation: str = "",
) -> tuple[str, str]:
    if eval_name == "conversation-recall":
        if not conversation.strip():
            raise ValueError("conversation-recall scoring requires conversation history")
        return CONVERSATION_JUDGE_SYSTEM, (
            f"Conversation:\n{conversation}\n\n"
            f"Question: {question}\n"
            f"Transcript: {transcript}"
        )
    if eval_name == "fact-recall":
        if not gold_answer.strip():
            raise ValueError("fact-recall scoring requires a gold answer")
        return JUDGE_SYSTEM, (
            f"Question: {question}\n"
            f"Gold answer: {gold_answer}\n"
            f"Transcript: {transcript}"
        )
    raise ValueError(f"recall judging requires a recall eval, got {eval_name}")


def judge_correct(
    system: str,
    prompt: str,
    *,
    model: str = DEFAULT_JUDGE_MODEL,
    complete: Callable[..., str] | None = None,
) -> bool:
    finish = complete or openai_json_complete
    parsed = json.loads(finish(system, prompt, model=model))
    if not isinstance(parsed, dict):
        raise ValueError("recall judge must return a JSON object")
    return bool(parsed.get("correct"))


def rescore_recall_payload(
    data: dict,
    *,
    model: str,
    complete: Callable[..., str],
) -> dict:
    """Re-judge a stored recall score. Unscored rollouts are returned unchanged."""
    payload = dict(data)
    if str(data.get("status", SCORED)) != SCORED or data.get("score") is None:
        return payload
    eval_name = str(data.get("eval", ""))
    if eval_name not in RECALL_EVAL_TYPES:
        raise ValueError(f"recall rescore requires a recall eval, got {eval_name}")
    system, prompt = recall_judge_request(
        eval_name,
        question=str(data.get("question", "")),
        transcript=str(data.get("transcript", "")),
        gold_answer=str(data.get("gold_answer", "")),
        conversation=str(data.get("conversation", "")),
    )
    correct = judge_correct(system, prompt, model=model, complete=complete)
    payload["score"] = 1 if correct else 0
    payload["judge_model"] = model
    return payload


def write_score_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def rescore_recall_file(
    source: Path,
    dest: Path,
    *,
    model: str,
    complete: Callable[..., str],
) -> str:
    if dest.is_file():
        return "skipped"
    data = json.loads(source.read_text(encoding="utf-8"))
    if str(data.get("status", SCORED)) != SCORED or data.get("score") is None:
        write_score_json(dest, data)
        return "copied"
    write_score_json(dest, rescore_recall_payload(data, model=model, complete=complete))
    return "judged"


def rescore_recall_tree(
    run_dir: Path,
    output_dir: Path,
    *,
    model: str,
    complete: Callable[..., str],
    workers: int = 16,
) -> dict[str, int]:
    """Write a parallel recall tree. Source scores are not modified."""
    counts = {"judged": 0, "copied": 0, "skipped": 0}
    jobs: list[Path] = []
    for eval_name in RECALL_EVAL_TYPES:
        root = run_dir / eval_name
        if not root.is_dir():
            continue
        for source in sorted(root.glob("**/rollout_*/error.json")):
            dest = output_dir / source.relative_to(run_dir)
            if dest.is_file():
                counts["skipped"] += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.read_bytes())
            counts["copied"] += 1
        jobs.extend(sorted(root.glob("**/rollout_*/score.json")))

    def rescore(source: Path) -> str:
        outcome = rescore_recall_file(
            source,
            output_dir / source.relative_to(run_dir),
            model=model,
            complete=complete,
        )
        print(f"{outcome} {source.relative_to(run_dir)}", flush=True)
        return outcome

    if not jobs:
        return counts
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for outcome in pool.map(rescore, jobs):
            counts[outcome] += 1
    return counts


def score_recall(
    trace: RolloutTrace,
    *,
    question: str,
    history_target_sec: float,
    history_sec: float,
    first_word_sec: float | None,
    transcript: str = "",
    gold_answer: str = "",
    conversation: str = "",
) -> RecallScore:
    if trace.expected_action not in RECALL_ACTIONS:
        raise ValueError(
            f"recall scoring requires conversation_recall or fact_recall, "
            f"got {trace.expected_action}"
        )
    eval_type = ACTION_EVAL_TYPES[trace.expected_action]
    scored = trace.status == SCORED
    if scored:
        system, prompt = recall_judge_request(
            eval_type,
            question=question,
            transcript=transcript,
            gold_answer=gold_answer,
            conversation=conversation,
        )
        correct = judge_correct(system, prompt)
    else:
        correct = False
    latency_ms, wordless_phonation = eot_onset(
        first_word_sec,
        trace.events,
        trace.checkpoint_sec,
        post_checkpoint_intervals(
            trace.played_audio, trace.events, trace.checkpoint_sec
        ),
    )
    return RecallScore(
        eval=eval_type,
        status=trace.status,
        score=(1 if correct else 0) if scored else None,
        question=question,
        gold_answer=gold_answer,
        transcript=transcript,
        history_target_sec=history_target_sec,
        history_sec=history_sec,
        timed_out=trace.timed_out,
        conversation=conversation,
        latency_ms=latency_ms,
        wordless_phonation=wordless_phonation,
    )
