"""Live Opus streaming session after optional duplex prefill."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np
import sphn
import torch
from aiohttp import web

from bench.audio import FrameAccumulator, append_pcm16_buffer
from prefill import PrefillDiagnostics, run_duplex_prefill
from text_alignment import EPAD_TOKEN, PAD_TOKEN
from wire import (
    MSG_ASSISTANT_HISTORY,
    MSG_AUDIO,
    MSG_COMMIT,
    MSG_HANDSHAKE,
    MSG_META,
    MSG_PRIMED,
    MSG_TEXT,
    MSG_TRANSCRIPT,
    MSG_USER_HISTORY,
)

logger = logging.getLogger(__name__)

# A peer that vanishes without a close frame would otherwise hold the global
# model lock until the container times out, blocking every later rollout.
SESSION_IDLE_TIMEOUT_SEC = 120.0
MAX_WS_MSG_SIZE = 128 * 1024**2


async def open_session_socket(request: web.Request) -> web.WebSocketResponse:
    # heartbeat=None: aiohttp only observes a pong from inside ws.receive(), and
    # nothing reads the socket during duplex prefill, so a heartbeat closes every
    # history longer than heartbeat * 1.5 however promptly the client pongs. The
    # idle watchdog above is what covers a vanished peer.
    ws = web.WebSocketResponse(max_msg_size=MAX_WS_MSG_SIZE, heartbeat=None)
    await ws.prepare(request)
    return ws


@dataclass
class DuplexSessionBuffers:
    user_pcm: np.ndarray | None = None
    assistant_pcm: np.ndarray | None = None
    words: list[dict] = field(default_factory=list)


async def run_live_session(
    state,
    ws: web.WebSocketResponse,
    voice_prompt_filename: str | None,
    text_prompt: str,
    seed: int | None,
    prefill: DuplexSessionBuffers | None,
) -> web.WebSocketResponse:
    close = False
    pending_messages: list = []
    live_audio_packets = 0
    last_message_at = time.monotonic()

    async def recv_loop(opus_reader) -> None:
        nonlocal close, live_audio_packets, last_message_at
        try:
            while True:
                if pending_messages:
                    message = pending_messages.pop(0)
                else:
                    message = await ws.receive()
                last_message_at = time.monotonic()
                if message.type == aiohttp.WSMsgType.ERROR:
                    break
                if message.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE):
                    break
                if message.type != aiohttp.WSMsgType.BINARY:
                    continue
                payload = message.data
                if not isinstance(payload, bytes) or not payload:
                    continue
                kind = payload[0]
                if kind != MSG_AUDIO:
                    continue
                live_audio_packets += 1
                opus_reader.append_bytes(payload[1:])
        finally:
            close = True

    async def opus_loop(opus_reader, opus_writer) -> None:
        frames = FrameAccumulator(state.frame_size)
        while True:
            if close:
                return
            await asyncio.sleep(0.001)
            pcm = opus_reader.read_pcm()
            if pcm.shape[-1] == 0:
                continue
            for chunk in frames.push(pcm):
                # A backlog is drained one blocking step per frame, so without
                # a yield the loop stalls for the whole backlog at once.
                await asyncio.sleep(0)
                chunk_tensor = torch.from_numpy(chunk)
                chunk_tensor = chunk_tensor.to(device=state.device)[None, None]
                codes = state.mimi.encode(chunk_tensor)
                for code_idx in range(codes.shape[-1]):
                    tokens = state.lm_gen.step(codes[:, :, code_idx : code_idx + 1])
                    if tokens is None:
                        continue
                    if tokens.shape[1] != state.lm_gen.lm_model.dep_q + 1:
                        raise RuntimeError(
                            f"expected {state.lm_gen.lm_model.dep_q + 1} codebooks, "
                            f"got {tokens.shape[1]}"
                        )
                    main_pcm = state.mimi.decode(tokens[:, 1:9])
                    opus_writer.append_pcm(main_pcm.cpu()[0, 0].numpy())
                    text_token = tokens[0, 0, 0].item()
                    if text_token not in (EPAD_TOKEN, PAD_TOKEN):
                        piece = state.text_tokenizer.id_to_piece(text_token)
                        piece = piece.replace("▁", " ")
                        await ws.send_bytes(bytes([MSG_TEXT]) + piece.encode("utf-8"))

    async def send_loop(opus_writer) -> None:
        while True:
            if close:
                return
            await asyncio.sleep(0.001)
            msg = opus_writer.read_bytes()
            if msg:
                await ws.send_bytes(bytes([MSG_AUDIO]) + msg)

    async def watchdog() -> None:
        while not close:
            idle = time.monotonic() - last_message_at
            if idle > SESSION_IDLE_TIMEOUT_SEC:
                logger.warning("closing session idle for %.0fs", idle)
                return
            await asyncio.sleep(min(1.0, SESSION_IDLE_TIMEOUT_SEC))

    async with state.lock:
        if seed is not None and seed != -1:
            state.apply_seed(seed)

        voice_prompt_path = state.resolve_voice_prompt(voice_prompt_filename)
        state.mimi.reset_streaming()
        state.other_mimi.reset_streaming()
        state.lm_gen.reset_streaming()
        state.configure_session(voice_prompt_path, text_prompt)

        async def is_alive() -> bool:
            if close or ws.closed:
                return False
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=0.01)
            except asyncio.TimeoutError:
                return True
            except aiohttp.ClientConnectionError:
                return False
            if msg.type in (
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.ERROR,
            ):
                return False
            if msg.type == aiohttp.WSMsgType.BINARY:
                pending_messages.append(msg)
            return True

        await state.lm_gen.step_system_prompts_async(state.mimi, is_alive=is_alive)
        state.mimi.reset_streaming()

        if prefill is not None:
            if prefill.user_pcm is None or prefill.assistant_pcm is None:
                raise RuntimeError("duplex prefill requires user and assistant audio")
            diagnostics: PrefillDiagnostics = await run_duplex_prefill(
                state.mimi,
                state.other_mimi,
                state.lm_gen,
                state.text_tokenizer,
                prefill.user_pcm,
                prefill.assistant_pcm,
                prefill.words,
                state.device,
                state.frame_size,
                max_history_frames=getattr(state, "max_history_frames", None),
            )
            primed_payload = json.dumps(
                {
                    "frame_count": diagnostics.frame_count,
                    "history_duration_sec": diagnostics.history_duration_sec,
                    "elapsed_sec": diagnostics.elapsed_sec,
                    "realtime_factor": diagnostics.realtime_factor,
                    "truncated_frames": diagnostics.truncated_frames,
                }
            ).encode("utf-8")
            await ws.send_bytes(bytes([MSG_PRIMED]) + primed_payload)
        else:
            await ws.send_bytes(bytes([MSG_HANDSHAKE]))

        # The watchdog measures how long the peer has been quiet, not how long
        # prefill took, and prefill of a long history can outlast the timeout.
        last_message_at = time.monotonic()
        opus_writer = sphn.OpusStreamWriter(state.mimi.sample_rate)
        opus_reader = sphn.OpusStreamReader(state.mimi.sample_rate)
        tasks = [
            asyncio.create_task(recv_loop(opus_reader)),
            asyncio.create_task(opus_loop(opus_reader, opus_writer)),
            asyncio.create_task(send_loop(opus_writer)),
            asyncio.create_task(watchdog()),
        ]
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in tasks:
            task.cancel()
        for result in await asyncio.gather(*tasks, return_exceptions=True):
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                # Otherwise a task that died mid-session leaves the client with
                # a quiet, healthy-looking connection and nothing in the logs.
                logger.error("session task failed", exc_info=result)

    await ws.close()
    return ws


async def run_duplex_session(state, ws: web.WebSocketResponse) -> web.WebSocketResponse:
    buffers = DuplexSessionBuffers()
    session_meta: dict | None = None

    while not ws.closed:
        message = await ws.receive()
        if message.type in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.ERROR,
        ):
            break
        if message.type != aiohttp.WSMsgType.BINARY:
            continue
        payload = message.data
        if not payload:
            continue
        kind = payload[0]
        body = payload[1:]
        if kind == MSG_META:
            session_meta = json.loads(body.decode("utf-8"))
        elif kind == MSG_USER_HISTORY:
            buffers.user_pcm = append_pcm16_buffer(buffers.user_pcm, body)
        elif kind == MSG_ASSISTANT_HISTORY:
            buffers.assistant_pcm = append_pcm16_buffer(buffers.assistant_pcm, body)
        elif kind == MSG_TRANSCRIPT:
            transcript = json.loads(body.decode("utf-8"))
            buffers.words = transcript.get("words", [])
        elif kind == MSG_COMMIT:
            if session_meta is None:
                raise RuntimeError("duplex commit received before metadata")
            return await run_live_session(
                state,
                ws,
                voice_prompt_filename=session_meta.get("voice_prompt"),
                text_prompt=session_meta.get("text_prompt", ""),
                seed=session_meta.get("seed"),
                prefill=buffers,
            )
        else:
            logger.warning("unexpected duplex control kind %s", kind)

    await ws.close()
    return ws
