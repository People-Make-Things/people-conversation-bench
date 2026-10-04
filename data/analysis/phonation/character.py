"""Acoustic character of the wordless above-gate audio.

A short falling "mm-hm" and a sustained flat-pitch vowel are both 110-135 Hz, so
dominant frequency alone decides nothing. What separates them is whether pitch
and spectrum move: real speech (and real backchannels) modulate f0 and shift
formants continuously, a drone holds one pitch and one filter.

Per phonation span: YIN-style autocorrelation pitch track, voiced fraction,
f0 median and its coefficient of variation, harmonic-to-noise ratio, LPC formant
estimates, spectral flux, and the span duration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from paths import ROOT  # noqa: E402

from detect import CELL_SEC, OUT, ROOT, analyze, load_cache, read, runs  # noqa: E402
from bench.audio_metrics import spectral_flatness, spectral_flux  # noqa: E402
from bench.transcribe import load_whisper_model  # noqa: E402

PITCH_WINDOW_SEC = 0.04
PITCH_HOP_SEC = 0.01
F0_MIN = 60.0
F0_MAX = 400.0
VOICED_HARMONICITY = 0.35
MIN_SPAN_SEC = 0.20


def pitch_track(pcm: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame f0 and normalized autocorrelation peak (a harmonicity proxy)."""
    window = int(PITCH_WINDOW_SEC * rate)
    hop = int(PITCH_HOP_SEC * rate)
    low, high = int(rate / F0_MAX), int(rate / F0_MIN)
    f0: list[float] = []
    strength: list[float] = []
    for offset in range(0, max(0, pcm.shape[0] - window) + 1, hop):
        frame = pcm[offset : offset + window].astype(np.float64)
        frame = frame - frame.mean()
        energy = float(np.dot(frame, frame))
        if energy <= 0:
            f0.append(0.0)
            strength.append(0.0)
            continue
        size = 1 << (2 * frame.shape[0] - 1).bit_length()
        spectrum = np.fft.rfft(frame, size)
        correlation = np.fft.irfft(spectrum * np.conj(spectrum), size)[: frame.shape[0]]
        top = min(high, correlation.shape[0] - 1)
        if low >= top:
            f0.append(0.0)
            strength.append(0.0)
            continue
        lag = int(np.argmax(correlation[low:top])) + low
        peak = float(correlation[lag] / energy)
        f0.append(rate / lag)
        strength.append(peak)
    return np.array(f0), np.array(strength)


def formants(pcm: np.ndarray, rate: int, order: int = 12) -> list[float]:
    """LPC pole frequencies; a vowel shows two clear resonances below 3 kHz."""
    signal = pcm.astype(np.float64)
    signal = signal - signal.mean()
    if signal.shape[0] < order * 4 or not np.any(signal):
        return []
    signal = np.append(signal[0], signal[1:] - 0.97 * signal[:-1])
    signal = signal * np.hamming(signal.shape[0])
    autocorrelation = np.correlate(signal, signal, mode="full")[signal.shape[0] - 1 :]
    if autocorrelation[0] <= 0:
        return []
    matrix = np.array(
        [autocorrelation[np.abs(np.arange(order) - index)] for index in range(order)]
    )
    try:
        coefficients = np.linalg.solve(
            matrix + np.eye(order) * autocorrelation[0] * 1e-6,
            autocorrelation[1 : order + 1],
        )
    # A near-singular autocorrelation matrix means this span carries no usable
    # resonance structure, which is a result rather than an error.
    except np.linalg.LinAlgError:
        return []
    roots = np.roots(np.concatenate([[1.0], -coefficients]))
    roots = roots[np.imag(roots) > 0]
    roots = roots[np.abs(roots) > 0.85]
    frequencies = np.sort(np.angle(roots) * rate / (2 * np.pi))
    return [float(value) for value in frequencies if 200 < value < 4000]


def span_report(pcm: np.ndarray, rate: int) -> dict:
    f0, strength = pitch_track(pcm, rate)
    voiced = strength >= VOICED_HARMONICITY
    voiced_f0 = f0[voiced]
    return {
        "duration_sec": pcm.shape[0] / rate,
        "rms": float(np.sqrt(np.mean(np.square(pcm.astype(np.float64))))),
        "voiced_fraction": float(voiced.mean()) if voiced.size else 0.0,
        "harmonicity_median": float(np.median(strength)) if strength.size else 0.0,
        "f0_median": float(np.median(voiced_f0)) if voiced_f0.size else 0.0,
        "f0_cv": (
            float(np.std(voiced_f0) / np.mean(voiced_f0))
            if voiced_f0.size > 1 and np.mean(voiced_f0) > 0
            else 0.0
        ),
        "f0_semitone_range": (
            float(12 * np.log2(voiced_f0.max() / voiced_f0.min()))
            if voiced_f0.size > 1 and voiced_f0.min() > 0
            else 0.0
        ),
        "f0_abs_slope_st_per_sec": (
            float(
                np.median(np.abs(np.diff(12 * np.log2(np.maximum(voiced_f0, 1e-9)))))
                / PITCH_HOP_SEC
            )
            if voiced_f0.size > 1
            else 0.0
        ),
        "spectral_flux": spectral_flux(pcm, rate),
        "spectral_flatness": spectral_flatness(pcm, rate),
        "formants": formants(pcm, rate)[:3],
    }


def collect(cohort: str, rows: list[dict], model, cache: dict, want_phonation: bool) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        pcm, rate = read(Path(row["path"]), row["start_sec"], row["end_sec"])
        analysis = analyze(model, row["key"], pcm, rate, cache)
        flags = analysis.phonation() if want_phonation else (
            analysis.gated() & analysis.labeled
        )
        for start, end in runs(flags):
            if (end - start) * CELL_SEC < MIN_SPAN_SEC:
                continue
            segment = pcm[int(start * CELL_SEC * rate) : int(end * CELL_SEC * rate)]
            if segment.size < int(0.05 * rate):
                continue
            entry = span_report(segment, rate)
            entry["cohort"] = cohort
            entry["key"] = row["key"]
            entry["start_sec"] = start * CELL_SEC
            out.append(entry)
    return out


def show(name: str, spans: list[dict]) -> None:
    if not spans:
        print(f"{name:34s} (no spans)")
        return
    def stat(field: str) -> str:
        values = np.array([s[field] for s in spans])
        return f"{np.median(values):7.3f}"
    durations = np.array([s["duration_sec"] for s in spans])
    print(
        f"{name:34s} n={len(spans):4d} "
        f"dur_med={np.median(durations):6.2f} dur_p90={np.percentile(durations, 90):6.2f} "
        f"dur_max={durations.max():6.2f} "
        f"voiced={stat('voiced_fraction')} harm={stat('harmonicity_median')} "
        f"f0={np.median([s['f0_median'] for s in spans]):6.1f} "
        f"f0cv={stat('f0_cv')} f0slope={stat('f0_abs_slope_st_per_sec')} "
        f"strange={np.median([s['f0_semitone_range'] for s in spans]):6.2f} "
        f"flux={stat('spectral_flux')} flat={stat('spectral_flatness')} "
        f"nform={np.median([len(s['formants']) for s in spans]):.1f}"
    )


def main() -> None:
    report = json.loads((OUT / "cohorts.json").read_text())
    cache = load_cache()
    model = load_whisper_model()
    results: dict[str, list[dict]] = {}
    for cohort in ("model_pause", "model_eot", "model_pause_prefix", "human_eot", "human_full"):
        results[f"{cohort}:wordless"] = collect(cohort, report[cohort], model, cache, True)
        results[f"{cohort}:lexical"] = collect(cohort, report[cohort], model, cache, False)
    (OUT / "character.json").write_text(json.dumps(results))

    print(
        "Spans >= 0.20 s. f0cv/f0slope/strange measure pitch movement; a sustained\n"
        "moan holds pitch (low cv, low slope, small semitone range), speech does not.\n"
    )
    for name, spans in results.items():
        show(name, spans)


if __name__ == "__main__":
    main()
