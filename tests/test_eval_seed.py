"""Rollouts share one model process, so each needs its own reproducible seed."""

from __future__ import annotations

from pathlib import Path

from bench.eval import BASE_SEED, rollout_seed
from bench.interact import build_session_config
from bench.protocol import SessionConfig
from bench.registry import AudioConfig, ConnectionConfig, ModelConfig


def config(seed: int | None) -> ModelConfig:
    return ModelConfig(
        id="fake",
        name="Fake",
        model_dir=Path("."),
        adapter_path=Path("adapter.py"),
        connection=ConnectionConfig(url="fake"),
        audio=AudioConfig(
            input_sample_rate=24000,
            output_sample_rate=24000,
            encoding="opus",
            frame_samples=1920,
        ),
        session=SessionConfig(seed=seed),
    )


def session_seeds(model_config: ModelConfig, rollouts: int) -> list[int | None]:
    return [
        build_session_config(
            model_config,
            voice_prompt=None,
            text_prompt=None,
            seed=rollout_seed(model_config, index),
        ).seed
        for index in range(rollouts)
    ]


def test_rollouts_get_distinct_but_reproducible_seeds() -> None:
    assert session_seeds(config(None), 3) == [BASE_SEED, BASE_SEED + 1, BASE_SEED + 2]
    assert session_seeds(config(None), 3) == session_seeds(config(None), 3)


def test_model_toml_seed_overrides_the_base() -> None:
    assert session_seeds(config(1000), 2) == [1000, 1001]


def test_seed_of_minus_one_sends_no_seed() -> None:
    assert session_seeds(config(-1), 2) == [None, None]


def test_eval_uses_the_configured_voice_prompt() -> None:
    """Eval leaves voice_prompt to model.toml. PersonaPlex clears it there so
    the teacher-forced history is the voice; GPT and Gemini keep theirs."""
    model_config = config(None)
    model_config.session.voice_prompt = "alloy"

    session = build_session_config(
        model_config,
        voice_prompt=None,
        text_prompt="persona",
        seed=1,
    )

    assert session.voice_prompt == "alloy"
    assert session.text_prompt == "persona"


def test_empty_configured_voice_prompt_clears_the_stock_voice() -> None:
    model_config = config(None)
    model_config.session.voice_prompt = ""

    session = build_session_config(
        model_config,
        voice_prompt=None,
        text_prompt="persona",
        seed=1,
    )

    assert session.voice_prompt == ""
