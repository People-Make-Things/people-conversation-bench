"""Shared "above the RMS gate but no decodable word" detector.

Identical decode and label rules to the speech-gate calibration described in
AGENTS.md (2 s windows at a 1 s hop, each peak normalized so the labeler cannot
inherit the level bias being measured; a cell is speech only when every window
covering it puts a word over it). Applied here to arbitrary PCM so the model's
own response.wav and the human other-speaker channel go through the same code.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from bench.audio import read_wav_segment, write_wav_pcm16  # noqa: E402
from bench.audio_metrics import rms, rms_envelope  # noqa: E402
from bench.speech import (  # noqa: E402
    SPEECH_MIN_DURATION_SEC,
    SPEECH_RELEASE_RATIO,
    SPEECH_RMS_THRESHOLD,
    PlayedAudio,
    speech_intervals,
)
from bench.transcribe import load_whisper_model  # noqa: E402

OUT = ROOT / "data/analysis/phonation"
CACHE = OUT / "windows.json"
SCRATCH = OUT / "_window.wav"

CELL_SEC = 0.02
WINDOW_SEC = 2.0
HOP_SEC = 1.0
NORMALIZED_PEAK = 0.5

MAX_NO_SPEECH_PROB = 0.5
MIN_AVG_LOGPROB = -1.2
MAX_COMPRESSION_RATIO = 2.4


def harmonicity(pcm: np.ndarray, rate: int) -> float:
    if pcm.size < rate // 40:
        return 0.0
    signal = pcm.astype(np.float64) - float(np.mean(pcm))
    energy = float(np.dot(signal, signal))
    if energy <= 0:
        return 0.0
    size = 1 << (2 * signal.shape[0] - 1).bit_length()
    spectrum = np.fft.rfft(signal, size)
    correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[: signal.shape[0]]
    low, high = rate // 400, min(rate // 60, correlation.shape[0] - 1)
    if low >= high:
        return 0.0
    return float(np.max(correlation[low:high]) / energy)


def decode(model, pcm: np.ndarray, rate: int, initial_prompt: str | None = None) -> dict:
    peak = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    scaled = pcm * (NORMALIZED_PEAK / peak) if peak > 0 else pcm
    write_wav_pcm16(SCRATCH, scaled, rate)
    result = model.transcribe(
        str(SCRATCH),
        word_timestamps=True,
        language="en",
        temperature=0.0,
        condition_on_previous_text=False,
        no_speech_threshold=None,
        logprob_threshold=None,
        initial_prompt=initial_prompt,
        verbose=False,
    )
    segments = result["segments"]
    return {
        "text": result["text"],
        "peak": peak,
        "rms": rms(pcm),
        "harmonicity": harmonicity(pcm, rate),
        "no_speech_prob": min((s["no_speech_prob"] for s in segments), default=1.0),
        "avg_logprob": max((s["avg_logprob"] for s in segments), default=-10.0),
        "compression_ratio": max(
            (s["compression_ratio"] for s in segments), default=99.0
        ),
        "words": [
            {
                "text": word["word"].strip(),
                "start": word["start"],
                "end": word["end"],
                "probability": word.get("probability", 0.0),
            }
            for segment in segments
            for word in segment.get("words", [])
            if word["word"].strip()
        ],
    }


def tile(model, pcm: np.ndarray, rate: int, initial_prompt: str | None = None) -> list[dict]:
    window = int(WINDOW_SEC * rate)
    hop = int(HOP_SEC * rate)
    windows = []
    for offset in range(0, max(1, pcm.shape[0] - window + hop), hop):
        chunk = pcm[offset : offset + window]
        if chunk.size == 0:
            break
        entry = decode(model, chunk, rate, initial_prompt)
        entry["start_sec"] = offset / rate
        entry["end_sec"] = (offset + chunk.shape[0]) / rate
        windows.append(entry)
    return windows


def window_is_speech(window: dict) -> bool:
    return (
        window["no_speech_prob"] <= MAX_NO_SPEECH_PROB
        and window["avg_logprob"] >= MIN_AVG_LOGPROB
        and window["compression_ratio"] <= MAX_COMPRESSION_RATIO
        and any(word["end"] > word["start"] for word in window["words"])
    )


def truth_flags(windows: list[dict], centers: np.ndarray) -> np.ndarray:
    coverage = np.zeros(centers.shape[0], dtype=int)
    support = np.zeros(centers.shape[0], dtype=int)
    for window in windows:
        inside = (centers >= window["start_sec"]) & (centers < window["end_sec"])
        coverage += inside
        if not window_is_speech(window):
            continue
        spoken = np.zeros(centers.shape[0], dtype=bool)
        for word in window["words"]:
            if word["end"] <= word["start"]:
                continue
            start = window["start_sec"] + word["start"]
            end = window["start_sec"] + word["end"]
            spoken |= (centers >= start) & (centers < end)
        support += spoken & inside
    required = np.minimum(coverage, 2)
    return (support >= required) & (support >= 1)


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < flags.shape[0]:
        if not flags[index]:
            index += 1
            continue
        start = index
        while index < flags.shape[0] and flags[index]:
            index += 1
        spans.append((start, index))
    return spans


@dataclass
class Analysis:
    key: str
    duration_sec: float
    envelope: np.ndarray
    labeled: np.ndarray
    windows: list[dict]

    @property
    def cells(self) -> int:
        return self.envelope.shape[0]

    def gated(self, threshold: float = SPEECH_RMS_THRESHOLD) -> np.ndarray:
        """Hysteresis gate from bench.speech, projected back onto 20 ms cells."""
        blocks = [
            PlayedAudio(
                start_sec=index * CELL_SEC,
                end_sec=(index + 1) * CELL_SEC,
                rms=float(value),
            )
            for index, value in enumerate(self.envelope)
        ]
        flags = np.zeros(self.cells, dtype=bool)
        for start, end in speech_intervals(
            blocks,
            rms_threshold=threshold,
            min_duration_sec=SPEECH_MIN_DURATION_SEC,
            release_ratio=SPEECH_RELEASE_RATIO,
        ):
            begin = int(round(start / CELL_SEC))
            finish = int(round(end / CELL_SEC))
            flags[begin:finish] = True
        return flags

    def phonation(self, threshold: float = SPEECH_RMS_THRESHOLD) -> np.ndarray:
        """Cells the gate calls speech that carry no decodable word."""
        return self.gated(threshold) & ~self.labeled

    def summary(self, threshold: float = SPEECH_RMS_THRESHOLD) -> dict:
        gated = self.gated(threshold)
        phonation = self.phonation(threshold)
        spans = runs(phonation)
        return {
            "key": self.key,
            "duration_sec": self.duration_sec,
            "cells": self.cells,
            "gated_sec": float(gated.sum()) * CELL_SEC,
            "labeled_sec": float(self.labeled.sum()) * CELL_SEC,
            "phonation_sec": float(phonation.sum()) * CELL_SEC,
            "gated_fraction": float(gated.mean()) if self.cells else 0.0,
            "phonation_fraction_of_gated": (
                float(phonation.sum() / gated.sum()) if gated.sum() else 0.0
            ),
            "phonation_spans": len(spans),
            "phonation_span_durations": [
                (end - start) * CELL_SEC for start, end in spans
            ],
            "any_labeled_word": bool(self.labeled.any()),
            "gated_but_wordless_file": bool(gated.any() and not self.labeled.any()),
            "peak_cell_rms": float(self.envelope.max()) if self.cells else 0.0,
        }


def analyze(model, key: str, pcm: np.ndarray, rate: int, cache: dict) -> Analysis:
    if key not in cache:
        cache[key] = {
            "duration_sec": pcm.shape[0] / rate,
            "windows": tile(model, pcm, rate),
        }
        CACHE.write_text(json.dumps(cache))
        print(f"decoded {key} ({len(cache[key]['windows'])} windows)", flush=True)
    envelope = rms_envelope(pcm, rate, CELL_SEC)
    centers = np.arange(envelope.shape[0]) * CELL_SEC + CELL_SEC / 2
    return Analysis(
        key=key,
        duration_sec=pcm.shape[0] / rate,
        envelope=envelope,
        labeled=truth_flags(cache[key]["windows"], centers),
        windows=cache[key]["windows"],
    )


def analyze_cached(key: str, pcm: np.ndarray, rate: int, cache: dict) -> Analysis:
    """Same labels as `analyze`, without loading a whisper model.

    Every key the clip and voicing scripts touch is already in the decode cache,
    so they need no model; a missing key is a signal to rebuild the cache rather
    than something to decode silently with different settings.
    """
    if key not in cache:
        raise KeyError(f"{key} is not in {CACHE}; run rebuild_cache.py first")
    return analyze(None, key, pcm, rate, cache)


def load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.is_file() else {}


def read(path: Path, start_sec: float = 0.0, end_sec: float | None = None):
    pcm, rate = read_wav_segment(path)
    begin = int(round(start_sec * rate))
    finish = pcm.shape[0] if end_sec is None else int(round(end_sec * rate))
    return pcm[begin:finish], rate
