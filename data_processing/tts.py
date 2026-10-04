"""Clone a speaker from a reference clip and speak a prepared probe.

Required for prepare-eval (`uv sync --extra tts`) unless `--max-points`.
Tests inject `synthesize` and never load the model.
"""

from __future__ import annotations

import numpy as np

from bench.audio import TARGET_SAMPLE_RATE, pad_pcm_to_model_frames, resample

QWEN_TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
TTS_INSTALL_HINT = (
    "voice cloning requires qwen-tts. Install with: uv sync --extra tts"
)

_tts_model = None


def require_tts() -> None:
    try:
        import qwen_tts  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(TTS_INSTALL_HINT) from exc


def load_tts_model():
    global _tts_model
    if _tts_model is None:
        # Optional extra: qwen-tts is not a core install, so import only here.
        require_tts()
        import torch
        from qwen_tts import Qwen3TTSModel
        device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        _tts_model = Qwen3TTSModel.from_pretrained(
            QWEN_TTS_MODEL,
            device_map=device_map,
        )
    return _tts_model


def synthesize_clone(
    reference_pcm: np.ndarray,
    reference_rate: int,
    reference_text: str,
    prompt_text: str,
) -> np.ndarray:
    """Speak prompt_text in the reference speaker's voice, 24 kHz frame-padded."""
    model = load_tts_model()
    ref_text = reference_text.strip()
    wavs, sample_rate = model.generate_voice_clone(
        text=prompt_text,
        language="English",
        ref_audio=(reference_pcm, reference_rate),
        ref_text=ref_text or None,
        x_vector_only_mode=not bool(ref_text),
    )
    pcm = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
    pcm = resample(pcm, int(sample_rate), TARGET_SAMPLE_RATE)
    return pad_pcm_to_model_frames(pcm)
