"""Transcript windows and LLM judging for curated split events.

The rendering here is the judge's entire view of a candidate: if a detail is
not in the rendered window, the judge cannot use it. The same rendering is
used for human inspection so reviewed samples and judged samples match.

The judge runs on gate survivors only (events with a heuristic score) and its
verdict re-ranks selection: an event the judge scores zero is never selected,
however few candidates a session has. Judged files are rewritten in place, so
an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3

from data_processing.curation_report import write_judge_figures
from data_processing.ground_truth import channel_words_from_ground_truth
from data_processing.source_seamless import SEAMLESS_BUCKET, SESSIONS_PREFIX
from utils.openai import openai_json_complete

TURN_WORD_LIMIT = 100
RESUME_WORD_LIMIT = 25
OTHER_CONTEXT_WORD_LIMIT = 40

JUDGE_VERSION = 1
# Selection floor: a session with only weak candidates keeps fewer (or no)
# points rather than diluting the eval with mid-clause pauses and trivial
# exchanges that happened to top their session.
MIN_JUDGE_SCORE = 0.4
# gpt-4o-mini could not separate rhetorical questions from turn handoffs,
# which zeroes exactly the most tempting fair pauses; gpt-4o can.
DEFAULT_JUDGE_MODEL = "gpt-4o"
JUDGE_WORKERS = 16
JUDGE_ATTEMPTS = 3

TRANSCRIPT_NOTES = """\
Notes on the transcripts: they are automatic and messy. CJK characters
(such as these 呃 嗯 哈 对 係) are filler or laughter artifacts; treat them as
noise, not incoherence. Some recording sessions mix scripted stimulus
sentences that participants read aloud (for example "The massive boulder
blocked the path, making it difficult to pass."); a window dominated by such
disconnected readings is not natural conversation."""

PAUSE_JUDGE_SYSTEM = f"""\
You screen candidate moments from real two-person conversations for a
turn-taking benchmark. In each candidate, one speaker (SPEAKER) goes silent
mid-turn and later continues. A voice assistant playing the OTHER SPEAKER
fails the benchmark if it starts talking during that silence, so a candidate
is only fair if staying quiet is what a good conversation partner would do.

{TRANSCRIPT_NOTES}

Return JSON:
{{"coherent": true/false,
 "turn_handoff": true/false,
 "temptation": 0-10,
 "reason": "one short sentence"}}

coherent: the window reads as natural connected conversation.
turn_handoff: the LAST WORDS are a question or request addressed to the
listener that the listener is expected to answer. Use the continuation to
decide: if the speaker asks a question and then continues by answering or
extending it themselves (a rhetorical or thinking-aloud question, e.g. "do I
stay in New York?" followed by "Or do I just go someplace"), it is NOT a
handoff. If the question is aimed at the listener (e.g. "Steve Nash or Amar'e
Stoudemire?") and the continuation does not answer it, it IS a handoff.
temptation: judge ONLY from the LAST WORDS BEFORE THE SILENCE, not from how
complete the story so far feels. If the FINAL word is a conjunction, article,
preposition, subject pronoun, auxiliary ("and", "the", "to", "I'm", "was",
"But"), or ends with a comma, the speaker is audibly mid-clause: score 0-2
even when the sentence before that fragment was complete.
- 0-2: last words end mid-clause - on a conjunction, article, preposition,
  pronoun, or filler ("and", "the", "I'm", "like,", "you know", "But").
- 3-5: clause is grammatically complete but clearly sets up more (a list in
  progress, "so...", a dangling comparison).
- 6-8: a finished clause that could end the turn, though the wider context
  hints the speaker has more to say.
- 9-10: sounds fully final - story beat resolved, opinion delivered, nothing
  syntactically or pragmatically pending.

The best candidates are coherent, are NOT turn handoffs, and have high
temptation: the speaker sounds finished, but a good listener would still
wait."""

EOT_JUDGE_SYSTEM = f"""\
You screen candidate moments from real two-person conversations for a
response benchmark. In each candidate a USER TURN ends and a voice assistant
playing the other speaker must reply; its reply is scored against what the
real human said next (ASSISTANT REPLY).

{TRANSCRIPT_NOTES}

Return JSON:
{{"coherent": true/false,
 "genuine_handoff": true/false,
 "responsive_reference": true/false,
 "difficulty": 0-10,
 "reason": "one short sentence"}}

coherent: the window reads as natural connected conversation; false if the
window is dominated by scripted readings or the reply text is mostly
artifact tokens.
genuine_handoff: the user turn actually ends and yields the floor; false if
it is an interrupted or abandoned fragment that trails off mid-thought.
responsive_reference: the human ASSISTANT REPLY actually responds to the
user turn, rather than continuing an unrelated earlier thread.
difficulty: how much a good reply must engage with the specific content of
the conversation. 0 = a generic acknowledgment ("yeah, totally") would be a
fine reply, 10 = the reply must use specific details from the conversation.

The best candidates are coherent, genuine handoffs with a responsive
reference and high difficulty."""


def load_cached_words(cache_dir: Path, source_id: str) -> dict[str, list[dict]]:
    path = cache_dir / f"{source_id}.json"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        s3 = boto3.Session(profile_name="pmt").client("s3")
        s3.download_file(
            SEAMLESS_BUCKET,
            f"{SESSIONS_PREFIX}{source_id}/ground_truth.json",
            str(path),
        )
    with path.open(encoding="utf-8") as handle:
        return channel_words_from_ground_truth(json.load(handle))


def span_words(
    channel_words: dict[str, list[dict]],
    speaker: str,
    start_sec: float,
    end_sec: float,
) -> list[str]:
    return [
        word["text"]
        for word in channel_words.get(speaker, [])
        if word["start"] < end_sec and word["end"] > start_sec
    ]


def clipped(words: list[str], limit: int, from_end: bool) -> str:
    if len(words) <= limit:
        return " ".join(words)
    if from_end:
        return "... " + " ".join(words[-limit:])
    return " ".join(words[:limit]) + " ..."


def other_speaker(speaker: str) -> str:
    return "speaker_2" if speaker == "speaker_1" else "speaker_1"


def render_pause_window(event: dict, channel_words: dict[str, list[dict]]) -> str:
    features = event["features"]
    checkpoint = event["timestamp"]
    turn_start = checkpoint - features["context_sec"]
    window_end = checkpoint + features["pause_sec"]
    active = event["active_speaker"]
    partner = other_speaker(active)

    partner_before = span_words(channel_words, partner, checkpoint - 20.0, checkpoint)
    turn = span_words(channel_words, active, turn_start, checkpoint)
    resume = span_words(channel_words, active, window_end, window_end + 10.0)
    return (
        f"OTHER SPEAKER (recent): "
        f"{clipped(partner_before, OTHER_CONTEXT_WORD_LIMIT, from_end=True)}\n"
        f"SPEAKER (turn so far): {clipped(turn, TURN_WORD_LIMIT, from_end=True)}\n"
        f"<<SPEAKER GOES SILENT FOR {features['pause_sec']:.1f} SECONDS>>\n"
        f"SPEAKER (after the silence): "
        f"{clipped(resume, RESUME_WORD_LIMIT, from_end=False)}\n"
        f"LAST WORDS BEFORE THE SILENCE: \"{' '.join(turn[-8:])}\""
    )


def render_eot_window(event: dict, channel_words: dict[str, list[dict]]) -> str:
    features = event["features"]
    eot = event["timestamp"]
    turn_start = eot - features["user_turn_sec"]
    reference_start = eot + features["gap_sec"]
    reference_end = reference_start + features["reference_sec"]
    active = event["active_speaker"]
    partner = other_speaker(active)

    partner_before = span_words(channel_words, partner, turn_start - 20.0, turn_start)
    turn = span_words(channel_words, active, turn_start, eot)
    reference = span_words(channel_words, partner, reference_start, reference_end)
    return (
        f"ASSISTANT SPEAKER (previous turn): "
        f"{clipped(partner_before, OTHER_CONTEXT_WORD_LIMIT, from_end=True)}\n"
        f"USER TURN (ends here, {features['user_turn_sec']:.1f}s): "
        f"{clipped(turn, TURN_WORD_LIMIT, from_end=True)}\n"
        f"ASSISTANT REPLY (human reference, {features['reference_sec']:.1f}s "
        f"after {features['gap_sec']:.1f}s gap): "
        f"{clipped(reference, TURN_WORD_LIMIT, from_end=False)}"
    )


def render_window(event: dict, channel_words: dict[str, list[dict]]) -> str:
    if event["split_type"] == "pause_start":
        return render_pause_window(event, channel_words)
    return render_eot_window(event, channel_words)


def scale_int(value) -> int:
    """Clamp a judge 0-10 value that may arrive as int, float, or string."""
    return max(0, min(10, int(float(value or 0))))


def pause_verdict(parsed: dict) -> dict:
    verdict = {
        "coherent": bool(parsed.get("coherent")),
        "turn_handoff": bool(parsed.get("turn_handoff")),
        "temptation": scale_int(parsed.get("temptation")),
        "reason": str(parsed.get("reason", "")),
    }
    fair = verdict["coherent"] and not verdict["turn_handoff"]
    verdict["score"] = verdict["temptation"] / 10 if fair else 0.0
    return verdict


def eot_verdict(parsed: dict) -> dict:
    verdict = {
        "coherent": bool(parsed.get("coherent")),
        "genuine_handoff": bool(parsed.get("genuine_handoff")),
        "responsive_reference": bool(parsed.get("responsive_reference")),
        "difficulty": scale_int(parsed.get("difficulty")),
        "reason": str(parsed.get("reason", "")),
    }
    fair = (
        verdict["coherent"]
        and verdict["genuine_handoff"]
        and verdict["responsive_reference"]
    )
    verdict["score"] = verdict["difficulty"] / 10 if fair else 0.0
    return verdict


def judge_event(
    event: dict,
    channel_words: dict[str, list[dict]],
    model: str = DEFAULT_JUDGE_MODEL,
) -> dict:
    window = render_window(event, channel_words)
    is_pause = event["split_type"] == "pause_start"
    system = PAUSE_JUDGE_SYSTEM if is_pause else EOT_JUDGE_SYSTEM
    for attempt in range(JUDGE_ATTEMPTS):
        # Retry transient API failures: 429 / 5xx surface as RuntimeError from
        # openai_json_complete, but a socket timeout while reading the response
        # body escapes that wrapper as OSError. One flaky call must not abort a
        # multi-thousand-call batch.
        try:
            parsed = json.loads(openai_json_complete(system, window, model=model))
            break
        except (RuntimeError, OSError):
            if attempt == JUDGE_ATTEMPTS - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
    verdict = pause_verdict(parsed) if is_pause else eot_verdict(parsed)
    return {"version": JUDGE_VERSION, "model": model, **verdict}


def judge_rank(event: dict) -> tuple[float, float]:
    """Judge score first, heuristic score as tiebreak."""
    judge = event.get("judge") or {}
    return (judge.get("score", 0.0), event.get("score") or 0.0)


def reselect(payload: dict) -> None:
    """Re-rank gate survivors by judge verdict and reapply the per-type cap.

    Unlike heuristic-only selection, an event below MIN_JUDGE_SCORE is never
    selected: quality over quota.
    """
    keep_per_type = payload["keep_per_type"]
    for split_type in ("pause_start", "end_of_turn"):
        scored = [
            event
            for event in payload["events"]
            if event["split_type"] == split_type and event["score"] is not None
        ]
        scored.sort(key=judge_rank, reverse=True)
        for rank, event in enumerate(scored):
            judge_score = (event.get("judge") or {}).get("score", 0.0)
            if judge_score < MIN_JUDGE_SCORE:
                event["keep"] = False
                event["reason"] = "judge_rejected"
            elif rank < keep_per_type:
                event["keep"] = True
                event["reason"] = "selected"
            else:
                event["keep"] = False
                event["reason"] = "outranked"


def needs_judging(event: dict, model: str) -> bool:
    if event["score"] is None:
        return False
    judge = event.get("judge")
    return (
        judge is None
        or judge.get("version") != JUDGE_VERSION
        or judge.get("model") != model
    )


def judge_curation(
    curation_dir: str | Path,
    cache_dir: str | Path = "data/curation_cache",
    model: str = DEFAULT_JUDGE_MODEL,
    workers: int = JUDGE_WORKERS,
) -> None:
    curation_path = Path(curation_dir)
    cache_path = Path(cache_dir)
    paths = sorted(curation_path.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no curation files under {curation_path}")

    judged_total = 0
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        pending = [
            event for event in payload["events"] if needs_judging(event, model)
        ]
        if pending:
            channel_words = load_cached_words(cache_path, payload["source_id"])
            with ThreadPoolExecutor(max_workers=workers) as pool:
                verdicts = list(
                    pool.map(
                        lambda event: judge_event(event, channel_words, model),
                        pending,
                    )
                )
            for event, verdict in zip(pending, verdicts):
                event["judge"] = verdict
            judged_total += len(pending)
        reselect(payload)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        kept = sum(1 for event in payload["events"] if event["keep"])
        survivors = sum(
            1 for event in payload["events"] if event["score"] is not None
        )
        print(
            f"{payload['source_id']}: judged {len(pending)}, "
            f"kept {kept} of {survivors} gate survivors",
            flush=True,
        )
    figure = write_judge_figures(curation_path)
    print(
        f"Done. Judged {judged_total} event(s) across {len(paths)} file(s). "
        f"Figures: {figure}"
    )
