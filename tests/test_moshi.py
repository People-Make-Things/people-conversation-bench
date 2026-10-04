"""Moshi is Kyutai Moshi registered on the shared PersonaPlex duplex path."""

from __future__ import annotations

import ast
import asyncio
import json
from pathlib import Path

import numpy as np
import prefill
import pytest
import websockets

from bench.audio import MODEL_FRAME_SAMPLES, pcm16_bytes_to_float, pcm_float_to_int16_bytes
from bench.protocol import ContextType, SessionConfig
from bench.registry import (
    discover_model_configs,
    load_adapter,
    load_model_config,
    resolve_model_ref,
)
from models.personaplex.adapter import build_ws_url
from models.personaplex.opus_codec import bind_missing_method
from models.personaplex.server.wire import MSG_PRIMED, MSG_TEXT

ROOT = Path(__file__).resolve().parent.parent
MOSHI_DEPLOY = ROOT / "models" / "moshi" / "deploy" / "serve.py"
DUPLEX_SERVER = ROOT / "models" / "personaplex" / "server" / "duplex_server.py"


class ChattyServer:
    def __init__(self) -> None:
        self.recv_calls = 0

    async def recv(self) -> bytes:
        self.recv_calls += 1
        return bytes([MSG_PRIMED]) + json.dumps({"history_duration_sec": 1.0}).encode()


def moshi_config():
    return load_model_config(resolve_model_ref("moshi"))


def test_moshi_is_registered_as_duplex() -> None:
    assert "moshi" in discover_model_configs()
    config = moshi_config()
    model = load_adapter(config)

    assert config.id == "moshi"
    assert config.name == "Kyutai Moshi 7B"
    assert config.max_concurrency == 48
    assert config.audio.encoding == "opus"
    assert config.audio.frame_samples == MODEL_FRAME_SAMPLES
    assert config.audio.input_sample_rate == 24000
    assert config.connection.url.endswith("people-bench-moshi-serve.modal.run")
    assert model.supported_contexts == (ContextType.DUPLEX_AUDIO,)


def test_moshi_shares_the_personaplex_adapter() -> None:
    config = moshi_config()
    model = load_adapter(config)

    assert config.adapter_path == (ROOT / "models" / "personaplex" / "adapter.py").resolve()
    assert type(model).__name__ == "PersonaPlexModel"
    assert type(model).__module__.endswith(".adapter")


def test_moshi_deploy_filename_does_not_shadow_the_moshi_package() -> None:
    """Modal mounts the entrypoint at /root/<name>.py. PYTHONPATH includes
    /root, so a file named moshi.py is imported instead of moshi.models."""
    assert MOSHI_DEPLOY.name != "moshi.py"
    assert not (MOSHI_DEPLOY.parent / "moshi.py").exists()
    assert 'Path("/root/moshi.py")' in MOSHI_DEPLOY.read_text(encoding="utf-8")


def test_moshi_deploy_serves_official_kyutai_weights() -> None:
    source = MOSHI_DEPLOY.read_text(encoding="utf-8")
    tree = ast.parse(source)
    namespace: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "server_command":
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(MOSHI_DEPLOY), "exec"), namespace)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MOSHI_HF_REPO":
                    exec(compile(ast.Module(body=[node], type_ignores=[]), str(MOSHI_DEPLOY), "exec"), namespace)

    command = namespace["server_command"]("/usr/bin/python", 8998)
    assert namespace["MOSHI_HF_REPO"] == "kyutai/moshiko-pytorch-bf16"
    assert command[command.index("--hf-repo") + 1] == "kyutai/moshiko-pytorch-bf16"
    assert command[command.index("--voice-prompt-dir") + 1] == "none"
    assert command[command.index("--static") + 1] == "none"


def test_duplex_server_skips_missing_config_and_voices() -> None:
    source = DUPLEX_SERVER.read_text(encoding="utf-8")
    assert 'hf_hub_download(args.hf_repo, "config.json")' not in source
    assert 'if voice_prompt_dir == "none":' in source
    assert "def load_moshi_lm(" in source
    assert '"personaplex" in hf_repo' in source
    assert "max_history_frames_for_repo(args.hf_repo)" in source


def test_prefill_cap_follows_official_context_windows() -> None:
    assert prefill.PERSONAPLEX_MAX_HISTORY_FRAMES == 2048
    assert prefill.MOSHI_MAX_HISTORY_FRAMES == 3000
    assert (
        prefill.max_history_frames_for_repo("nvidia/personaplex-7b-v1")
        == prefill.PERSONAPLEX_MAX_HISTORY_FRAMES
    )
    assert (
        prefill.max_history_frames_for_repo("kyutai/moshiko-pytorch-bf16")
        == prefill.MOSHI_MAX_HISTORY_FRAMES
    )


def test_wait_for_primed_does_not_spin_on_a_message_it_cannot_use() -> None:
    model = load_adapter(moshi_config())
    early = bytes([MSG_TEXT]) + b"hello"
    model._early_messages = [early]
    server = ChattyServer()

    asyncio.run(asyncio.wait_for(model._wait_for_primed(server), timeout=5))

    assert server.recv_calls == 1
    assert model._early_messages == [early]


def test_bind_missing_method_raises_instead_of_binding_silence() -> None:
    class Renamed:
        def get_bytes(self) -> bytes:
            return b"ok"

    class Missing:
        pass

    assert bind_missing_method(Renamed(), "read_bytes", ("get_bytes",))() == b"ok"
    with pytest.raises(AttributeError, match="unsupported"):
        bind_missing_method(Missing(), "read_bytes", ("get_bytes",))


def test_trailing_model_audio_is_flushed_on_close() -> None:
    model = load_adapter(moshi_config())
    model._frames.push(np.full(MODEL_FRAME_SAMPLES // 2, 0.5, dtype=np.float32))

    async def run() -> np.ndarray:
        await model._flush_opus_frames()
        return (await model._audio_queue.get()).pcm

    flushed = asyncio.run(run())
    assert flushed.shape[0] == MODEL_FRAME_SAMPLES
    assert np.count_nonzero(flushed) == MODEL_FRAME_SAMPLES // 2


def test_pcm16_round_trip_is_symmetric() -> None:
    pcm = np.linspace(-1.0, 1.0, 4001, dtype=np.float32)
    restored = pcm16_bytes_to_float(pcm_float_to_int16_bytes(pcm))

    assert np.max(np.abs(restored - pcm)) <= 1.0 / 32768.0


def test_preload_url_drops_the_query_string() -> None:
    session = SessionConfig(voice_prompt="", text_prompt="p" * 9000, seed=3)
    url = build_ws_url("http://host", session, path="/api/duplex", include_session=False)

    assert url == "ws://host/api/duplex"
    assert len(url) < 8190


def test_connection_error_is_latched_for_the_rollout_to_see() -> None:
    model = load_adapter(moshi_config())
    assert model.connection_error is None

    error = websockets.ConnectionClosedError(None, None)
    model._io_error = error
    assert model.connection_error is error
