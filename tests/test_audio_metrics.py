"""Calibration tests for the objective audio metrics.

These pin down what each metric reads on signals whose character is known, so
other tests and the rollout analyzer can rely on the thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.audio_metrics import (
    SPEECH_RMS,
    best_lag_correlation,
    clipping_ratio,
    describe,
    discontinuity_ratio,
    dominant_frequency,
    envelope_similarity,
    rms,
    rms_envelope,
    speech_intervals,
    speech_ratio,
    spectral_flatness,
    spectral_flux,
)

SAMPLE_RATE = 24000


def seconds(count: float) -> np.ndarray:
    return np.arange(int(SAMPLE_RATE * count)) / SAMPLE_RATE


def tone(duration_sec: float, frequency: float = 220.0, amplitude: float = 0.2) -> np.ndarray:
    return (amplitude * np.sin(2 * np.pi * frequency * seconds(duration_sec))).astype(
        np.float32
    )


def noise(duration_sec: float, amplitude: float = 0.2) -> np.ndarray:
    generator = np.random.default_rng(0)
    return (amplitude * generator.standard_normal(seconds(duration_sec).size)).astype(
        np.float32
    )


def speech_like(duration_sec: float) -> np.ndarray:
    """Amplitude and frequency modulated, the way voiced speech moves."""
    time = seconds(duration_sec)
    envelope = (0.5 + 0.5 * np.sin(2 * np.pi * 3.5 * time)) ** 2
    carrier = np.sin(2 * np.pi * (180 + 120 * np.sin(2 * np.pi * 2.1 * time)) * time)
    return (0.3 * envelope * carrier).astype(np.float32)


def test_rms_and_envelope_track_level() -> None:
    assert rms(np.zeros(1000, dtype=np.float32)) == 0.0
    assert rms(tone(1.0)) == pytest.approx(0.2 / np.sqrt(2), abs=0.01)
    envelope = rms_envelope(tone(1.0), SAMPLE_RATE)
    assert envelope.shape[0] == 50
    assert np.all(envelope > SPEECH_RMS)


def test_silence_reads_as_silence() -> None:
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    assert speech_ratio(silence, SAMPLE_RATE) == 0.0
    assert speech_intervals(silence, SAMPLE_RATE) == []
    assert spectral_flatness(silence, SAMPLE_RATE) == 0.0
    assert describe(silence, SAMPLE_RATE).rms == 0.0


def test_speech_intervals_find_the_gaps() -> None:
    signal = speech_like(2.0)
    signal[int(0.8 * SAMPLE_RATE) : int(1.2 * SAMPLE_RATE)] = 0
    intervals = speech_intervals(signal, SAMPLE_RATE)
    assert intervals
    assert not any(start < 1.15 < end for start, end in intervals)
    assert 0.0 < speech_ratio(signal, SAMPLE_RATE) < 1.0


def test_spectral_flatness_separates_tones_from_noise() -> None:
    assert spectral_flatness(tone(1.0), SAMPLE_RATE) < 0.05
    assert spectral_flatness(noise(1.0), SAMPLE_RATE) > 0.3


def test_spectral_flux_separates_a_drone_from_moving_speech() -> None:
    # A held tone is loud but spectrally frozen, which is what the analyzer
    # treats as a drone rather than speech.
    assert spectral_flux(tone(2.0), SAMPLE_RATE) < 0.05
    assert spectral_flux(speech_like(2.0), SAMPLE_RATE) > 0.2


def test_dominant_frequency_recovers_the_tone() -> None:
    assert abs(dominant_frequency(tone(0.5, frequency=440.0), SAMPLE_RATE) - 440.0) < 5.0


def test_discontinuity_ratio_spots_a_spliced_out_frame() -> None:
    clean = tone(1.0)
    spliced = np.concatenate([clean[:12000], clean[12000 + 1921 :]])
    assert discontinuity_ratio(clean) < 0.1
    assert discontinuity_ratio(spliced) > 0.4


def test_cross_correlation_recovers_delay_and_rejects_unrelated_audio() -> None:
    signal = speech_like(1.0)
    delayed = np.concatenate([np.zeros(1000, dtype=np.float32), signal])[: signal.size]
    assert best_lag_correlation(signal, signal) == (0, pytest.approx(1.0, abs=0.001))
    lag, correlation = best_lag_correlation(delayed, signal)
    assert lag == 1000
    # Not 1.0: shifting the copy truncates its tail.
    assert correlation > 0.97
    _, unrelated = best_lag_correlation(signal, noise(1.0))
    assert unrelated < 0.2


def test_envelope_similarity_is_high_only_for_matching_envelopes() -> None:
    signal = speech_like(2.0)
    assert envelope_similarity(signal, signal, SAMPLE_RATE) > 0.99
    assert envelope_similarity(signal, noise(2.0), SAMPLE_RATE) < 0.3


def test_clipping_ratio_counts_rails() -> None:
    signal = np.concatenate(
        [np.full(100, 1.0, dtype=np.float32), np.zeros(900, dtype=np.float32)]
    )
    assert clipping_ratio(signal) == pytest.approx(0.1, abs=0.001)
    assert clipping_ratio(tone(1.0)) == 0.0


def test_describe_summarises_without_crashing_on_empty_audio() -> None:
    report = describe(np.zeros(0, dtype=np.float32), SAMPLE_RATE)
    assert report.duration_sec == 0.0
    assert report.to_dict()["peak"] == 0.0
