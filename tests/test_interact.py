from __future__ import annotations

import queue
import threading

import numpy as np

from bench.interact import InteractSession
from bench.protocol import AudioFrame


def make_session() -> InteractSession:
    session = InteractSession.__new__(InteractSession)
    session.output_interrupted = threading.Event()
    session.output_queue = queue.Queue()
    session._output_frame = None
    session._output_offset = 0
    return session


def test_output_tracks_partial_and_multiple_frames() -> None:
    session = make_session()
    first_played = []
    second_played = []
    session.output_queue.put(
        AudioFrame(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            first_played.append,
        )
    )
    session.output_queue.put(
        AudioFrame(
            np.array([0.6, 0.7], dtype=np.float32),
            second_played.append,
        )
    )

    first_output = np.zeros((3, 1), dtype=np.float32)
    session.on_audio_output(first_output, 3, None, None)
    second_output = np.zeros((4, 1), dtype=np.float32)
    session.on_audio_output(second_output, 4, None, None)

    np.testing.assert_allclose(first_output[:, 0], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(second_output[:, 0], [0.4, 0.5, 0.6, 0.7])
    assert first_played == [3, 2]
    assert second_played == [2]


def test_interruption_clears_unplayed_audio() -> None:
    session = make_session()
    played = []
    session._output_frame = AudioFrame(
        np.array([0.1, 0.2], dtype=np.float32),
        played.append,
    )
    session.output_queue.put(
        AudioFrame(
            np.array([0.3, 0.4], dtype=np.float32),
            played.append,
        )
    )

    session.interrupt_playback()
    output = np.ones((2, 1), dtype=np.float32)
    session.on_audio_output(output, 2, None, None)

    np.testing.assert_array_equal(output, np.zeros((2, 1), dtype=np.float32))
    assert session._output_frame is None
    assert session._output_offset == 0
    assert session.output_queue.empty()
    assert played == []
