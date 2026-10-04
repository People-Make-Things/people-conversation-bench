"""GPU-free stand-in for the PersonaPlex duplex server.

The real server coroutines (`run_live_session` / `run_duplex_session`) are driven
against fake Mimi and LMGen objects that echo user audio back one frame later.
Everything else on the path is real: aiohttp websockets opened with the
production socket options, the duplex wire protocol, the sphn Opus codec, and the
production client adapter. That makes the whole pipeline runnable locally with no
GPU, no Modal deploy, and no model weights, while still failing when the server
loses or reorders audio.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from aiohttp import web

from bench.audio import MODEL_FRAME_SAMPLES, TARGET_SAMPLE_RATE
from bench.protocol import SessionConfig
from bench.registry import AudioConfig, ConnectionConfig, ModelConfig
from live_session import open_session_socket, run_duplex_session, run_live_session
from text_alignment import EPAD_TOKEN, PAD_TOKEN

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "personaplex"

# PersonaPlex text stream specials; both are filtered out of the model's output.
TEXT_PAD = PAD_TOKEN
TEXT_EPAD = EPAD_TOKEN
SPOKEN_TOKEN = 1234
TEXT_EVERY_N_FRAMES = 4
# PersonaPlex decodes tokens[:, 1:9], so the LM emits text plus this many.
AUDIO_CODEBOOKS = 8


class FakeMimi:
    """Store every encoded frame and hand it back verbatim on decode."""

    def __init__(self, sample_rate: int = TARGET_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.encoded: list[np.ndarray] = []
        self.decoded: list[np.ndarray] = []
        self.resets = 0

    def reset_streaming(self) -> None:
        self.resets += 1

    def encode(self, chunk: torch.Tensor) -> torch.Tensor:
        self.encoded.append(chunk[0, 0].cpu().numpy().copy())
        return torch.tensor([[[len(self.encoded) - 1]]], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        pcm = self.encoded[int(tokens[0, 0, 0].item())]
        self.decoded.append(pcm)
        return torch.from_numpy(pcm)[None, None]

    def received_pcm(self) -> np.ndarray:
        if not self.encoded:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.encoded)


class FakeLMGen:
    """Echo the frame it was handed and emit a text token now and then."""

    def __init__(self, step_delay_sec: float = 0.0, prefill_delay_sec: float = 0.0):
        self.step_delay_sec = step_delay_sec
        self.prefill_delay_sec = prefill_delay_sec
        self.lm_model = SimpleNamespace(dep_q=AUDIO_CODEBOOKS)
        self.live_steps = 0
        self.prefill_steps = 0
        self.prefill_text_tokens: list[int] = []
        self.live_step_times: list[float] = []
        self.resets = 0
        self.system_prompt_calls = 0
        self.text_prompt_tokens: list[int] = []
        self.voice_prompt_path: str | None = None

    def reset_streaming(self) -> None:
        self.resets += 1

    async def step_system_prompts_async(self, mimi, is_alive=None) -> None:
        self.system_prompt_calls += 1
        if is_alive is not None:
            await is_alive()

    def load_voice_prompt_embeddings(self, path: str) -> None:
        self.voice_prompt_path = path

    def clear_voice_prompt(self) -> None:
        self.voice_prompt_path = None

    def step(self, codes, other_codes=None, text_token=None):
        if other_codes is not None:
            self.prefill_steps += 1
            self.prefill_text_tokens.append(int(text_token.reshape(-1)[0].item()))
            if self.prefill_delay_sec:
                time.sleep(self.prefill_delay_sec)
        else:
            self.live_steps += 1
            self.live_step_times.append(time.monotonic())
            if self.step_delay_sec:
                time.sleep(self.step_delay_sec)
        speaking = self.live_steps % TEXT_EVERY_N_FRAMES == 0
        text = SPOKEN_TOKEN if speaking and other_codes is None else TEXT_PAD
        frame_id = int(codes[0, 0, 0].item())
        return torch.tensor(
            [[[text]] + [[frame_id]] * AUDIO_CODEBOOKS], dtype=torch.long
        )


class FakeTextTokenizer:
    """One token per character, so multi-piece words are the normal case."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def id_to_piece(self, index: int) -> str:
        return "\u2581ok" if index == SPOKEN_TOKEN else chr(index)


@dataclass
class FakeServerState:
    mimi: FakeMimi = field(default_factory=FakeMimi)
    other_mimi: FakeMimi = field(default_factory=FakeMimi)
    lm_gen: FakeLMGen = field(default_factory=FakeLMGen)
    text_tokenizer: FakeTextTokenizer = field(default_factory=FakeTextTokenizer)
    frame_size: int = MODEL_FRAME_SAMPLES
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    seeds: list[int] = field(default_factory=list)
    voice_prompts: list[str | None] = field(default_factory=list)
    text_prompts: list[str] = field(default_factory=list)

    def apply_seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def resolve_voice_prompt(self, voice_prompt_filename: str | None) -> str | None:
        return voice_prompt_filename or None

    def configure_session(self, voice_prompt_path: str | None, text_prompt: str) -> None:
        self.voice_prompts.append(voice_prompt_path)
        self.text_prompts.append(text_prompt)
        # Mirrors ServerState.configure_session: the process is reused across
        # sessions, so a session without a prompt must clear the previous one.
        if voice_prompt_path is None:
            self.lm_gen.clear_voice_prompt()
        else:
            self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
        self.lm_gen.text_prompt_tokens = (
            self.text_tokenizer.encode(text_prompt) if text_prompt else []
        )


@asynccontextmanager
async def serve(state: FakeServerState):
    """Run the real server handlers over loopback and yield the base URL."""

    async def handle_chat(request: web.Request) -> web.WebSocketResponse:
        ws = await open_session_socket(request)
        return await run_live_session(
            state,
            ws,
            voice_prompt_filename=request.query.get("voice_prompt"),
            text_prompt=request.query.get("text_prompt", ""),
            seed=int(request.query["seed"]) if "seed" in request.query else None,
            prefill=None,
        )

    async def handle_duplex(request: web.Request) -> web.WebSocketResponse:
        return await run_duplex_session(state, await open_session_socket(request))

    app = web.Application(client_max_size=128 * 1024**2)
    app.router.add_get("/api/chat", handle_chat)
    app.router.add_get("/api/duplex", handle_duplex)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        host, port = runner.addresses[0][:2]
        yield f"http://{host}:{port}"
    finally:
        await runner.cleanup()


def model_config(base_url: str, session: SessionConfig | None = None) -> ModelConfig:
    model_dir = MODEL_DIR
    return ModelConfig(
        id="personaplex-loopback",
        name="PersonaPlex loopback",
        model_dir=model_dir,
        adapter_path=model_dir / "adapter.py",
        connection=ConnectionConfig(url=base_url),
        audio=AudioConfig(
            input_sample_rate=TARGET_SAMPLE_RATE,
            output_sample_rate=TARGET_SAMPLE_RATE,
            encoding="opus",
            frame_samples=MODEL_FRAME_SAMPLES,
        ),
        session=session or SessionConfig(voice_prompt="NATF2.pt"),
    )


def frame_ladder(frame_count: int, frame_samples: int = MODEL_FRAME_SAMPLES) -> np.ndarray:
    """Frames of distinct, bin-aligned tones so losses are locatable after Opus.

    Frame i holds exactly ``16 + i`` periods inside the frame, which puts it on
    its own FFT bin and starts and ends at zero phase, so concatenation adds no
    discontinuity of its own.
    """
    phase = 2 * np.pi * np.arange(frame_samples) / frame_samples
    return np.concatenate(
        [0.3 * np.sin((16 + index) * phase) for index in range(frame_count)]
    ).astype(np.float32)


def ladder_frequencies(frame_count: int, sample_rate: int = TARGET_SAMPLE_RATE) -> list[float]:
    frame_rate = sample_rate / MODEL_FRAME_SAMPLES
    return [(16 + index) * frame_rate for index in range(frame_count)]
