"""Objective audio measurements so rollouts can be verified without listening.

Every metric here is a plain function over mono float PCM. They exist so tests
and triage tooling can assert on audio ("this window is silent", "what came back
matches what we sent", "this output is a static drone") instead of asking a
human to play a WAV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bench.speech import (
    SPEECH_MAX_GAP_SEC,
    SPEECH_MIN_DURATION_SEC,
    SPEECH_RELEASE_RATIO,
    SPEECH_RMS_THRESHOLD,
    PlayedAudio,
    speech_intervals as gated_speech_intervals,
)

SPEECH_RMS = SPEECH_RMS_THRESHOLD
ANALYSIS_WINDOW_SEC = 0.02
EPSILON = 1e-12


def rms(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(pcm, dtype=np.float64))))


def frame_signal(pcm: np.ndarray, window_samples: int) -> np.ndarray:
    """Split pcm into non-overlapping windows, dropping any partial tail."""
    count = pcm.shape[0] // window_samples
    if count == 0:
        return np.zeros((0, window_samples), dtype=np.float64)
    return pcm[: count * window_samples].astype(np.float64).reshape(count, window_samples)


def rms_envelope(
    pcm: np.ndarray,
    sample_rate: int,
    window_sec: float = ANALYSIS_WINDOW_SEC,
) -> np.ndarray:
    windows = frame_signal(pcm, max(1, int(sample_rate * window_sec)))
    if windows.shape[0] == 0:
        return np.zeros(0)
    return np.sqrt(np.mean(np.square(windows), axis=1))


def played_audio(
    pcm: np.ndarray,
    sample_rate: int,
    window_sec: float = ANALYSIS_WINDOW_SEC,
) -> list[PlayedAudio]:
    envelope = rms_envelope(pcm, sample_rate, window_sec)
    return [
        PlayedAudio(index * window_sec, (index + 1) * window_sec, float(value))
        for index, value in enumerate(envelope)
    ]


def speech_intervals(
    pcm: np.ndarray,
    sample_rate: int,
    rms_threshold: float = SPEECH_RMS,
    min_duration_sec: float = SPEECH_MIN_DURATION_SEC,
    max_gap_sec: float = SPEECH_MAX_GAP_SEC,
    window_sec: float = ANALYSIS_WINDOW_SEC,
) -> list[tuple[float, float]]:
    """Same hysteresis gate scoring uses, built from a raw PCM envelope."""
    return gated_speech_intervals(
        played_audio(pcm, sample_rate, window_sec),
        rms_threshold=rms_threshold,
        min_duration_sec=min_duration_sec,
        max_gap_sec=max_gap_sec,
        release_ratio=SPEECH_RELEASE_RATIO,
    )


def speech_ratio(
    pcm: np.ndarray,
    sample_rate: int,
    rms_threshold: float = SPEECH_RMS,
    window_sec: float = ANALYSIS_WINDOW_SEC,
) -> float:
    duration = pcm.shape[0] / sample_rate if sample_rate else 0.0
    if duration <= 0:
        return 0.0
    intervals = speech_intervals(
        pcm, sample_rate, rms_threshold=rms_threshold, window_sec=window_sec
    )
    return sum(end - start for start, end in intervals) / duration


def power_spectra(
    pcm: np.ndarray,
    sample_rate: int,
    window_sec: float = ANALYSIS_WINDOW_SEC,
) -> np.ndarray:
    window_samples = max(1, int(sample_rate * window_sec))
    windows = frame_signal(pcm, window_samples)
    if windows.shape[0] == 0:
        return np.zeros((0, window_samples // 2 + 1))
    tapered = windows * np.hanning(window_samples)
    return np.abs(np.fft.rfft(tapered, axis=1)) ** 2


def spectral_flatness(
    pcm: np.ndarray,
    sample_rate: int,
    window_sec: float = ANALYSIS_WINDOW_SEC,
) -> float:
    """Wiener entropy: near 1 for broadband noise, near 0 for tonal audio.

    Silent windows are excluded because their spectrum is numerical floor, which
    would otherwise read as perfectly flat.
    """
    spectra = power_spectra(pcm, sample_rate, window_sec)
    spectra = spectra[np.mean(spectra, axis=1) > EPSILON] if spectra.shape[0] else spectra
    if spectra.shape[0] == 0:
        return 0.0
    geometric = np.exp(np.mean(np.log(spectra + EPSILON), axis=1))
    arithmetic = np.mean(spectra, axis=1) + EPSILON
    return float(np.mean(geometric / arithmetic))


def spectral_flux(
    pcm: np.ndarray,
    sample_rate: int,
    window_sec: float = ANALYSIS_WINDOW_SEC,
) -> float:
    """Mean frame-to-frame spectral change; low means a static, droning sound."""
    spectra = power_spectra(pcm, sample_rate, window_sec)
    if spectra.shape[0] < 2:
        return 0.0
    normalized = spectra / (np.sum(spectra, axis=1, keepdims=True) + EPSILON)
    return float(np.mean(np.sum(np.abs(np.diff(normalized, axis=0)), axis=1)) / 2)


def dominant_frequency(pcm: np.ndarray, sample_rate: int) -> float:
    if pcm.shape[0] < 2:
        return 0.0
    spectrum = np.abs(np.fft.rfft(pcm.astype(np.float64) * np.hanning(pcm.shape[0])))
    return float(np.fft.rfftfreq(pcm.shape[0], 1 / sample_rate)[int(np.argmax(spectrum))])


def discontinuity_ratio(pcm: np.ndarray) -> float:
    """Largest sample-to-sample jump relative to peak level.

    A spliced-out chunk of audio leaves a step edge, so this rises far above
    what a continuously sampled waveform can produce.
    """
    if pcm.shape[0] < 2:
        return 0.0
    peak = float(np.max(np.abs(pcm)))
    if peak <= EPSILON:
        return 0.0
    return float(np.max(np.abs(np.diff(pcm.astype(np.float64)))) / peak)


def clipping_ratio(pcm: np.ndarray, threshold: float = 0.999) -> float:
    if pcm.size == 0:
        return 0.0
    return float(np.mean(np.abs(pcm) >= threshold))


def best_lag_correlation(
    reference: np.ndarray,
    candidate: np.ndarray,
    max_lag_samples: int | None = None,
) -> tuple[int, float]:
    """Peak normalized cross-correlation and the lag (in samples) that achieves it."""
    if reference.size == 0 or candidate.size == 0:
        return 0, 0.0
    length = reference.shape[0] + candidate.shape[0]
    size = 1 << (length - 1).bit_length()
    left = np.fft.rfft(reference.astype(np.float64), size)
    right = np.fft.rfft(candidate.astype(np.float64), size)
    correlation = np.fft.irfft(left * np.conj(right), size)
    correlation = np.concatenate([correlation[-(candidate.shape[0] - 1):], correlation[: reference.shape[0]]])
    lags = np.arange(-(candidate.shape[0] - 1), reference.shape[0])
    if max_lag_samples is not None:
        keep = np.abs(lags) <= max_lag_samples
        correlation = correlation[keep]
        lags = lags[keep]
    scale = np.linalg.norm(reference) * np.linalg.norm(candidate)
    if scale <= EPSILON or correlation.size == 0:
        return 0, 0.0
    index = int(np.argmax(correlation))
    return int(lags[index]), float(correlation[index] / scale)


def envelope_similarity(
    reference: np.ndarray,
    candidate: np.ndarray,
    sample_rate: int,
    window_sec: float = ANALYSIS_WINDOW_SEC,
) -> float:
    """Pearson correlation of RMS envelopes, tolerant of codec phase changes."""
    left = rms_envelope(reference, sample_rate, window_sec)
    right = rms_envelope(candidate, sample_rate, window_sec)
    length = min(left.shape[0], right.shape[0])
    if length < 2:
        return 0.0
    left = left[:length] - np.mean(left[:length])
    right = right[:length] - np.mean(right[:length])
    scale = np.linalg.norm(left) * np.linalg.norm(right)
    if scale <= EPSILON:
        return 0.0
    return float(np.dot(left, right) / scale)


@dataclass(frozen=True)
class AudioReport:
    duration_sec: float
    rms: float
    peak: float
    speech_ratio: float
    spectral_flatness: float
    spectral_flux: float
    discontinuity_ratio: float
    clipping_ratio: float

    def to_dict(self) -> dict:
        return {
            "duration_sec": self.duration_sec,
            "rms": self.rms,
            "peak": self.peak,
            "speech_ratio": self.speech_ratio,
            "spectral_flatness": self.spectral_flatness,
            "spectral_flux": self.spectral_flux,
            "discontinuity_ratio": self.discontinuity_ratio,
            "clipping_ratio": self.clipping_ratio,
        }


def describe(pcm: np.ndarray, sample_rate: int) -> AudioReport:
    return AudioReport(
        duration_sec=pcm.shape[0] / sample_rate,
        rms=rms(pcm),
        peak=float(np.max(np.abs(pcm))) if pcm.size else 0.0,
        speech_ratio=speech_ratio(pcm, sample_rate),
        spectral_flatness=spectral_flatness(pcm, sample_rate),
        spectral_flux=spectral_flux(pcm, sample_rate),
        discontinuity_ratio=discontinuity_ratio(pcm),
        clipping_ratio=clipping_ratio(pcm),
    )
