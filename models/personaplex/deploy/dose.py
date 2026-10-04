"""Modal GPU experiment: mask voiced-but-unworded prefill frames."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from rank_probe import (
    CONTINUATION_SEC,
    download_report,
    run_rank_probe,
    stage_histories,
    timed_word_dicts,
)

app = modal.App("people-bench-personaplex-dose")

DOSE_DIR = "dose_probe"
DOSE_POINTS = "000003,000006,000010,000014,000016,000021,000025"


def mask_assistant_history(
    assistant_pcm, sample_rate: int, words, arm: str
) -> tuple[object, float]:
    """Ablate voiced assistant history frames on the LM's own 80 ms frame grid.

    `mask_unworded` removes the frames that are above the speech gate with no
    stored word over them, which is the dose `prefill_gaps.py` measures.
    `mask_worded` removes the same number of voiced frames, spread evenly over
    frames a word does cover, so a result cannot be explained by having simply
    fed the model less audio. It does leave those words with no audio under them,
    which is a different inconsistency rather than a neutral one.
    """
    import numpy as np

    from bench.audio import MODEL_FRAME_SAMPLES
    from bench.audio_metrics import rms_envelope
    from bench.speech import SPEECH_RMS_THRESHOLD

    frames = assistant_pcm.shape[0] // MODEL_FRAME_SAMPLES
    frame_sec = MODEL_FRAME_SAMPLES / sample_rate
    voiced = rms_envelope(assistant_pcm, sample_rate, frame_sec)[:frames] >= (
        SPEECH_RMS_THRESHOLD
    )
    starts = np.arange(frames) * frame_sec
    worded = np.zeros(frames, dtype=bool)
    for word in words:
        end_sec = word.end_sec if word.end_sec is not None else word.start_sec + frame_sec
        worded |= (starts + frame_sec > word.start_sec) & (starts < end_sec)

    target = np.zeros(frames, dtype=bool)
    if arm == "mask_unworded":
        target = voiced & ~worded
    elif arm == "mask_worded":
        count = int((voiced & ~worded).sum())
        candidates = np.flatnonzero(voiced & worded)
        if count and candidates.size:
            picked = np.linspace(0, candidates.size - 1, count).round().astype(int)
            target[candidates[np.unique(picked)]] = True
    elif arm != "control":
        raise ValueError(f"unknown mask arm {arm}")

    masked = assistant_pcm.copy()
    for index in np.flatnonzero(target):
        masked[index * MODEL_FRAME_SAMPLES : (index + 1) * MODEL_FRAME_SAMPLES] = 0.0
    return masked, float(target.sum()) * frame_sec


@app.local_entrypoint()
def dose(
    points: str = DOSE_POINTS,
    manifest_dir: str = "data/processed/example_1/points",
    arms: str = "control,mask_unworded,mask_worded",
    seeds: int = 4,
    continuation_sec: float = CONTINUATION_SEC,
    out: str = "data/analysis/dose/dose_probe.json",
):
    """Does removing voiced-but-unworded assistant history shorten the warm-up?

    One point per distinct duplex history, each primed unmasked and with the dose
    removed, plus an arm that removes the same amount of voiced audio from frames
    a word does cover. Several seeds per cell, because the continuation is where
    the outcome is measured and it is the only stochastic part.
    """
    from bench.points import load_rollout_scenario
    from bench.protocol import ContextType

    entries = []
    for point in points.split(","):
        scenario = load_rollout_scenario(
            Path(manifest_dir) / point.strip(), kinds=(ContextType.DUPLEX_AUDIO,)
        )
        context = scenario.contexts[ContextType.DUPLEX_AUDIO]
        for arm in arms.split(","):
            assistant_pcm, masked_sec = mask_assistant_history(
                context.assistant_pcm, context.sample_rate, context.words, arm.strip()
            )
            entries.append(
                {
                    "dir": f"{scenario.point_id}_{arm.strip()}",
                    "history": "full",
                    "arm": arm.strip(),
                    "masked_sec": masked_sec,
                    "point_id": scenario.point_id,
                    "text_prompt": scenario.text_prompt,
                    "words": timed_word_dicts(context.words),
                    "user_pcm": context.user_pcm,
                    "assistant_pcm": assistant_pcm,
                    "sample_rate": context.sample_rate,
                }
            )
            print(
                f"{scenario.point_id} {arm.strip():14s} "
                f"history {context.assistant_pcm.shape[0] / context.sample_rate:6.2f} s, "
                f"masked {masked_sec:5.2f} s"
            )

    summary = run_rank_probe.remote(
        histories=stage_histories(DOSE_DIR, entries),
        voice_prompts=[None],
        continuation_sec=continuation_sec,
        seeds=seeds,
        label=DOSE_DIR,
    )
    print(json.dumps(summary, indent=2))
    print(download_report(DOSE_DIR, out))
