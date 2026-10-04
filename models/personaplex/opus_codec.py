"""Opus codec wrappers with version-tolerant sphn bindings."""

from __future__ import annotations

import numpy as np
import sphn


def bind_missing_method(obj, name: str, alternatives: tuple[str, ...]):
    """Accept a renamed sphn method, but never silently bind nothing.

    These are the only paths audio travels on. A no-op fallback turns an
    incompatible sphn into a rollout of pure silence that still scores.
    """
    if hasattr(obj, name):
        return getattr(obj, name)
    for alt in alternatives:
        if hasattr(obj, alt):
            return getattr(obj, alt)
    raise AttributeError(
        f"{type(obj).__name__} has no {name} and none of {alternatives}; "
        f"the installed sphn {getattr(sphn, '__version__', '')} is unsupported"
    )


class OpusStreams:
    def __init__(self, input_sample_rate: int, output_sample_rate: int) -> None:
        self.writer = sphn.OpusStreamWriter(input_sample_rate)
        self.reader = sphn.OpusStreamReader(output_sample_rate)
        self._read_bytes = bind_missing_method(
            self.writer,
            "read_bytes",
            ("get_bytes", "flush_bytes", "read_data"),
        )
        self._read_pcm = bind_missing_method(
            self.reader,
            "read_pcm",
            ("get_pcm", "receive_pcm", "read_float"),
        )

    def append_pcm(self, pcm: np.ndarray) -> None:
        self.writer.append_pcm(pcm.astype(np.float32, copy=False))

    def drain_encoded(self) -> bytes:
        encoded = self._read_bytes()
        return encoded or b""

    def append_encoded(self, payload: bytes) -> None:
        self.reader.append_bytes(payload)

    def read_pcm_frame(self) -> np.ndarray:
        # sphn returns None once the stream is closed and drained, and an empty
        # array when it is merely idle.
        pcm = self._read_pcm()
        if pcm is None or pcm.size == 0:
            return np.empty(0, np.float32)
        return pcm.astype(np.float32, copy=False)
