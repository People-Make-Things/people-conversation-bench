"""Word-aligned Whisper transcription for prepare-eval and scoring."""

from __future__ import annotations

import threading
from pathlib import Path

from bench.protocol import TimedWord

WHISPER_INSTALL_HINT = (
    "word-aligned transcription requires openai-whisper. "
    "Install with: uv sync"
)

_whisper_model = None
_whisper_lock = threading.Lock()


def require_whisper() -> None:
    try:
        import whisper  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(WHISPER_INSTALL_HINT) from exc


def load_whisper_model():
    with _whisper_lock:
        return _get_whisper_model()


def transcribe_words(wav_path: Path) -> list[TimedWord]:
    with _whisper_lock:
        model = _get_whisper_model()
        result = model.transcribe(
            str(wav_path),
            word_timestamps=True,
            language="en",
            verbose=False,
        )
    words: list[TimedWord] = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            text = str(word.get("word", "")).strip()
            if not text:
                continue
            words.append(
                TimedWord(
                    text=text,
                    start_sec=float(word["start"]),
                    end_sec=float(word["end"]),
                )
            )
    return words


def _get_whisper_model():
    """Return the process-wide Whisper model. Caller must hold `_whisper_lock`."""
    global _whisper_model
    if _whisper_model is None:
        require_whisper()
        import whisper

        _whisper_model = whisper.load_model("base")
    return _whisper_model
