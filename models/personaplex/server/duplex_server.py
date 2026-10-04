"""PersonaPlex duplex-preload WebSocket server."""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import sentencepiece
import torch
from aiohttp import web
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from live_session import open_session_socket, run_duplex_session, run_live_session
from prefill import PERSONAPLEX_MAX_HISTORY_FRAMES, max_history_frames_for_repo
from moshi.models import LMGen, LMModel, MimiModel, loaders
from moshi.utils.connection import create_ssl_context, get_lan_ip
from moshi.utils.logging import setup_logger

logger = setup_logger(__name__)
DeviceString = Literal["cuda"] | Literal["cpu"]


def wrap_with_system_tags(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
        return cleaned
    return f"<system> {cleaned} <system>"


def torch_auto_device(requested: DeviceString | None = None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


@dataclass
class ServerState:
    mimi: MimiModel
    other_mimi: MimiModel
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock
    device: torch.device
    voice_prompt_dir: str | None
    frame_size: int
    max_history_frames: int

    def apply_seed(self, seed: int) -> None:
        seed_all(seed)

    def __init__(
        self,
        mimi: MimiModel,
        other_mimi: MimiModel,
        text_tokenizer: sentencepiece.SentencePieceProcessor,
        lm: LMModel,
        device: str | torch.device,
        voice_prompt_dir: str | None = None,
        max_history_frames: int | None = None,
    ):
        self.mimi = mimi
        self.other_mimi = other_mimi
        self.text_tokenizer = text_tokenizer
        self.device = torch.device(device)
        self.voice_prompt_dir = voice_prompt_dir
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.max_history_frames = (
            PERSONAPLEX_MAX_HISTORY_FRAMES
            if max_history_frames is None
            else max_history_frames
        )
        self.lm_gen = LMGen(
            lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=self.device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
        )
        self.lock = asyncio.Lock()
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self) -> None:
        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for code_idx in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, code_idx : code_idx + 1])
                if tokens is None:
                    continue
                self.mimi.decode(tokens[:, 1:9])
                self.other_mimi.decode(tokens[:, 1:9])
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def resolve_voice_prompt(self, voice_prompt_filename: str | None) -> str | None:
        if self.voice_prompt_dir is None or not voice_prompt_filename:
            return None
        voice_prompt_path = os.path.join(self.voice_prompt_dir, voice_prompt_filename)
        if not os.path.exists(voice_prompt_path):
            raise FileNotFoundError(
                f"Requested voice prompt '{voice_prompt_filename}' not found in "
                f"'{self.voice_prompt_dir}'"
            )
        return voice_prompt_path

    def clear_voice_prompt(self) -> None:
        # LMGen exposes no way to unload a voice prompt, and the process is
        # reused across sessions, so a session without one would otherwise keep
        # speaking with the previous session's voice.
        self.lm_gen.voice_prompt = None
        self.lm_gen.voice_prompt_audio = None
        self.lm_gen.voice_prompt_embeddings = None
        self.lm_gen.voice_prompt_cache = None

    def configure_session(
        self,
        voice_prompt_path: str | None,
        text_prompt: str,
    ) -> None:
        if voice_prompt_path is None:
            self.clear_voice_prompt()
        elif voice_prompt_path.endswith(".pt"):
            self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
        else:
            self.lm_gen.load_voice_prompt(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
            if text_prompt
            else []
        )

    async def handle_chat(self, request: web.Request) -> web.WebSocketResponse:
        ws = await open_session_socket(request)
        logger.info("accepted stock /api/chat connection")
        return await run_live_session(
            self,
            ws,
            voice_prompt_filename=request.query.get("voice_prompt"),
            text_prompt=request.query.get("text_prompt", ""),
            seed=int(request.query["seed"]) if "seed" in request.query else None,
            prefill=None,
        )

    async def handle_duplex(self, request: web.Request) -> web.WebSocketResponse:
        ws = await open_session_socket(request)
        logger.info("accepted duplex /api/duplex connection")
        return await run_duplex_session(self, ws)


def copy_official_moshi_weights(state_dict: dict, model_sd: dict) -> dict:
    """Map official 8-codebook Moshi weights onto PersonaPlex's 16-codebook LM."""
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                state_dict[name] = torch.concat([tensor, tensor], dim=0)
    for name in model_sd:
        if name in state_dict:
            continue
        for old, new in zip(range(8), range(8, 16)):
            for prefix in ("gating", "linears", "depformer_in", "depformer_emb"):
                needle = f"{prefix}.{new}."
                if needle in name:
                    src = name.replace(needle, f"{prefix}.{old}.")
                    if src in state_dict:
                        state_dict[name] = state_dict[src]
                    break
    return state_dict


def materialize_meta_parameters(module: torch.nn.Module, device: torch.device) -> list[str]:
    leftover: list[str] = []
    for name, param in list(module.named_parameters()):
        if param.device.type != "meta":
            continue
        leftover.append(name)
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        parent.register_parameter(
            attr,
            torch.nn.Parameter(torch.zeros(param.shape, device=device, dtype=param.dtype)),
        )
    return leftover


def load_moshi_lm(moshi_weight: str, device: torch.device, hf_repo: str) -> LMModel:
    """Load PersonaPlex or official Kyutai weights into the PersonaPlex LM.

    Official checkpoints omit PersonaPlex-only parameters. NVIDIA's loader
    constructs those on the meta device and then ``Module.to()`` raises.
    """
    if "personaplex" in hf_repo:
        return loaders.get_moshi_lm(moshi_weight, device=device)

    lm = loaders.get_moshi_lm(None, device=torch.device("meta"))
    state = load_file(moshi_weight, device=device.type)
    state = copy_official_moshi_weights(state, lm.state_dict())
    for key, tensor in state.items():
        state[key] = tensor.to(device=device, dtype=torch.bfloat16)
    missing, unexpected = lm.load_state_dict(state, strict=False, assign=True)
    del state
    leftover = materialize_meta_parameters(lm, device)
    logger.info(
        "official moshi weights: missing=%s unexpected=%s leftover=%s",
        len(missing),
        len(unexpected),
        leftover,
    )
    return lm


def _get_voice_prompt_dir(voice_prompt_dir: str | None, hf_repo: str) -> str | None:
    if voice_prompt_dir == "none":
        return None
    if voice_prompt_dir is not None:
        return voice_prompt_dir
    voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
    voices_tgz = Path(voices_tgz)
    voices_dir = voices_tgz.parent / "voices"
    if not voices_dir.exists():
        with tarfile.open(voices_tgz, "r:gz") as tar:
            tar.extractall(path=voices_tgz.parent)
    if not voices_dir.exists():
        raise RuntimeError("voices.tgz did not contain a voices/ directory")
    return str(voices_dir)


def _get_static_path(static: str | None) -> str | None:
    if static is None:
        dist_tgz = hf_hub_download("nvidia/personaplex-7b-v1", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    if static == "none":
        return None
    return static


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--moshi-weight", type=str, default=None)
    parser.add_argument("--mimi-weight", type=str, default=None)
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--voice-prompt-dir", type=str, default=None)
    parser.add_argument("--ssl", type=str, default=None)
    args = parser.parse_args()

    args.voice_prompt_dir = _get_voice_prompt_dir(args.voice_prompt_dir, args.hf_repo)
    static_path = _get_static_path(args.static)
    device = torch_auto_device(args.device)
    seed_all(42424242)

    # Official Kyutai Moshi repos have no config.json; this server never reads
    # one. Weight / tokenizer downloads below are what fail a bad --hf-repo.
    if args.mimi_weight is None:
        args.mimi_weight = hf_hub_download(args.hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(args.mimi_weight, device)
    other_mimi = loaders.get_mimi(args.mimi_weight, device)

    if args.tokenizer is None:
        args.tokenizer = hf_hub_download(args.hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(args.tokenizer)

    if args.moshi_weight is None:
        args.moshi_weight = hf_hub_download(args.hf_repo, loaders.MOSHI_NAME)
    lm = load_moshi_lm(args.moshi_weight, device, args.hf_repo)
    lm.eval()

    state = ServerState(
        mimi=mimi,
        other_mimi=other_mimi,
        text_tokenizer=text_tokenizer,
        lm=lm,
        device=device,
        voice_prompt_dir=args.voice_prompt_dir,
        max_history_frames=max_history_frames_for_repo(args.hf_repo),
    )
    state.warmup()

    app = web.Application(client_max_size=128 * 1024**2)
    app.router.add_get("/api/chat", state.handle_chat)
    app.router.add_get("/api/duplex", state.handle_duplex)
    if static_path is not None:
        async def handle_root(_: web.Request) -> web.FileResponse:
            return web.FileResponse(os.path.join(static_path, "index.html"))

        app.router.add_get("/", handle_root)
        app.router.add_static("/", path=static_path, follow_symlinks=True, name="static")

    ssl_context = None
    protocol = "http"
    if args.ssl is not None:
        ssl_context, protocol = create_ssl_context(args.ssl)
    host_ip = args.host if args.host not in ("0.0.0.0", "::", "localhost") else get_lan_ip()
    logger.info("PersonaPlex duplex server at %s://%s:%s", protocol, host_ip, args.port)
    web.run_app(app, host=args.host, port=args.port, ssl_context=ssl_context)


if __name__ == "__main__":
    with torch.no_grad():
        main()
