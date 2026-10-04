"""Ablate the duplex history a step-driven model is primed with.

Used by GPU probes, not by `bench eval`. Trimming and dropping the teacher-forced
text stream belong with the experiment that varies them, not with loading a point.
"""

from __future__ import annotations

from dataclasses import replace

from bench.audio import MODEL_FRAME_SAMPLES
from bench.points import RolloutScenario, validate_history
from bench.protocol import ContextType, DuplexAudioContext, TimedWord


def rebase_words(
    words: tuple[TimedWord, ...],
    offset_sec: float,
    duration_sec: float,
) -> tuple[TimedWord, ...]:
    """Shift word times onto a history that lost `offset_sec` off its front."""
    shifted = []
    for word in words:
        start_sec = word.start_sec - offset_sec
        if start_sec < 0 or start_sec > duration_sec:
            continue
        end_sec = None if word.end_sec is None else word.end_sec - offset_sec
        shifted.append(TimedWord(text=word.text, start_sec=start_sec, end_sec=end_sec))
    return tuple(shifted)


def condition_duplex_history(
    scenario: RolloutScenario,
    history_sec: float | None = None,
    teacher_forced_text: bool = True,
) -> RolloutScenario:
    """Keep recent frames and/or drop the aligned word stream.

    `history_sec` keeps only the most recent whole frames of that much audio,
    dropping the oldest the way the per-model prefill cap already does
    server-side.
    Clearing `teacher_forced_text` sends the same audio with no word stream, so
    prefill teacher-forces all PAD instead of the aligned assistant transcript.
    """
    context = scenario.contexts.get(ContextType.DUPLEX_AUDIO)
    if context is None or (history_sec is None and teacher_forced_text):
        return scenario
    if not isinstance(context, DuplexAudioContext):
        raise TypeError(f"expected a duplex context, got {type(context).__name__}")

    user_pcm = context.user_pcm
    assistant_pcm = context.assistant_pcm
    offset_sec = 0.0
    if history_sec is not None:
        frames = max(1, int(history_sec * context.sample_rate) // MODEL_FRAME_SAMPLES)
        keep = min(user_pcm.shape[0], frames * MODEL_FRAME_SAMPLES)
        offset_sec = (user_pcm.shape[0] - keep) / context.sample_rate
        user_pcm = user_pcm[user_pcm.shape[0] - keep :]
        assistant_pcm = assistant_pcm[assistant_pcm.shape[0] - keep :]

    duration_sec = user_pcm.shape[0] / context.sample_rate
    words = (
        rebase_words(context.words, offset_sec, duration_sec)
        if teacher_forced_text
        else ()
    )
    validate_history(user_pcm, assistant_pcm, context.sample_rate, words)
    contexts = dict(scenario.contexts)
    contexts[ContextType.DUPLEX_AUDIO] = DuplexAudioContext(
        user_pcm=user_pcm,
        assistant_pcm=assistant_pcm,
        sample_rate=context.sample_rate,
        words=words,
    )
    return replace(scenario, contexts=contexts)
