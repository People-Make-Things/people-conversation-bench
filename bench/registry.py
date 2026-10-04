"""Discover model configs and instantiate adapters."""

from __future__ import annotations

import importlib.util
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import env

from bench.protocol import RealtimeModel, SessionConfig

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class AudioConfig:
    input_sample_rate: int
    output_sample_rate: int
    encoding: str
    frame_samples: int


SessionLayout = Literal["openai", "xai"]


@dataclass(frozen=True)
class ConnectionConfig:
    url: str | None = None
    url_env: str | None = None
    path: str | None = None
    api_key_env: str | None = None


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    model_dir: Path
    adapter_path: Path
    connection: ConnectionConfig
    audio: AudioConfig
    session: SessionConfig
    session_layout: SessionLayout = "openai"
    max_concurrency: int = 1


def resolve_connection(config: ModelConfig) -> str:
    connection = config.connection
    if connection.url:
        return connection.url

    base = env.get(connection.url_env) if connection.url_env else None
    if not base:
        env_key = connection.url_env or "connection URL"
        raise SystemExit(
            f"Model '{config.id}' needs {env_key}. Set it in the main checkout .env."
        )

    base = base.rstrip("/")
    if connection.path:
        return f"{base}{connection.path}"
    return base


def require_api_key(config: ModelConfig) -> str:
    name = config.connection.api_key_env
    if not name:
        raise SystemExit(
            f"Model '{config.id}' needs [connection] api_key_env in model.toml"
        )
    return env.require(name)


def discover_model_configs() -> dict[str, Path]:
    if not MODELS_DIR.is_dir():
        return {}

    configs: dict[str, Path] = {}
    for model_dir in sorted(MODELS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        config_path = model_dir / "model.toml"
        if config_path.is_file():
            data = tomllib.loads(config_path.read_text())
            model_id = data["model"]["id"]
            configs[model_id] = config_path
    return configs


def resolve_model_ref(model_ref: str) -> Path:
    if model_ref.startswith("@"):
        path = Path(model_ref[1:].strip())
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise SystemExit(f"Model config not found: {path}")
        return path

    configs = discover_model_configs()
    if model_ref not in configs:
        available = ", ".join(configs) if configs else "(none)"
        raise SystemExit(f"Unknown model '{model_ref}'. Available: {available}")
    return configs[model_ref]


def load_model_config(config_path: Path) -> ModelConfig:
    data = tomllib.loads(config_path.read_text())
    model_dir = config_path.parent

    model = data["model"]
    connection = data["connection"]
    audio = data["audio"]
    session = data.get("session", {})

    seed = session.get("seed")
    if seed is not None:
        seed = int(seed)

    session_extra = {
        key: str(value)
        for key, value in session.items()
        if key not in {"voice_prompt", "text_prompt", "seed"}
    }
    raw_layout = str(model.get("session_layout", "openai"))
    if raw_layout == "xai":
        session_layout: SessionLayout = "xai"
    elif raw_layout == "openai":
        session_layout = "openai"
    else:
        raise SystemExit(
            f"Model '{model['id']}' has unknown session_layout: {raw_layout}"
        )

    max_concurrency = int(model.get("max_concurrency", 1))
    if max_concurrency < 1:
        raise SystemExit(
            f"Model '{model['id']}' max_concurrency must be >= 1, got {max_concurrency}"
        )

    return ModelConfig(
        id=model["id"],
        name=model["name"],
        model_dir=model_dir,
        adapter_path=(model_dir / model.get("adapter", "adapter.py")).resolve(),
        connection=ConnectionConfig(
            url=connection.get("url"),
            url_env=connection.get("url_env"),
            path=connection.get("path"),
            api_key_env=connection.get("api_key_env"),
        ),
        audio=AudioConfig(
            input_sample_rate=int(audio["input_sample_rate"]),
            output_sample_rate=int(audio["output_sample_rate"]),
            encoding=audio["encoding"],
            frame_samples=int(audio["frame_samples"]),
        ),
        session=SessionConfig(
            voice_prompt=str(session.get("voice_prompt", "")),
            text_prompt=str(session.get("text_prompt", "")),
            seed=seed,
            extra=session_extra,
        ),
        session_layout=session_layout,
        max_concurrency=max_concurrency,
    )


def load_adapter(config: ModelConfig) -> RealtimeModel:
    adapter_path = config.adapter_path
    if not adapter_path.is_file():
        raise SystemExit(f"Missing adapter for model '{config.id}': {adapter_path}")

    pkg_name = f"people_bench_model_{config.id.replace('-', '_')}"
    if pkg_name not in sys.modules:
        pkg = type(sys)(pkg_name)
        pkg.__path__ = [str(adapter_path.parent)]
        sys.modules[pkg_name] = pkg

    module_name = f"{pkg_name}.adapter"
    spec = importlib.util.spec_from_file_location(module_name, adapter_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load adapter for model '{config.id}'")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    create = getattr(module, "create", None)
    if create is None:
        raise SystemExit(
            f"Adapter {adapter_path} must define create(config) -> RealtimeModel"
        )

    return create(config)


def load_model(model_ref: str) -> tuple[ModelConfig, RealtimeModel]:
    env.load()
    config_path = resolve_model_ref(model_ref)
    config = load_model_config(config_path)
    adapter = load_adapter(config)
    return config, adapter
