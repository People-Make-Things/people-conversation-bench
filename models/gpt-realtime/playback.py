"""Playback tracking and interruption handling for GPT Realtime audio."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass

import numpy as np
import websockets

from bench.protocol import AudioFrame, AudioInterrupted


@dataclass
class PlaybackState:
    response_id: str
    item_id: str
    content_index: int
    generation: int
    received_samples: int = 0
    played_samples: int = 0

    @property
    def key(self) -> tuple[str, str, int]:
        return self.response_id, self.item_id, self.content_index


class PlaybackTracker:
    def __init__(self, output_sample_rate: int, frame_samples: int) -> None:
        self.output_sample_rate = output_sample_rate
        self.frame_samples = frame_samples
        self._playback_lock = threading.Lock()
        self._receiving: PlaybackState | None = None
        self._playing: PlaybackState | None = None
        self._playback_generation = 0
        self._active_response_ids: set[str] = set()
        self._cancelled_response_ids: set[str] = set()
        self._pending_pcm = np.zeros(0, dtype=np.float32)

    @property
    def active_response_ids(self) -> set[str]:
        return self._active_response_ids

    @property
    def cancelled_response_ids(self) -> set[str]:
        return self._cancelled_response_ids

    def track_response_created(self, response_id: str | None) -> None:
        if response_id:
            self._active_response_ids.add(response_id)

    def track_response_done(self, response_id: str | None) -> None:
        if response_id:
            self._active_response_ids.discard(response_id)
            self._cancelled_response_ids.discard(response_id)

    def should_accept_response(self, response_id: str | None) -> bool:
        return (
            response_id is not None
            and response_id in self._active_response_ids
            and response_id not in self._cancelled_response_ids
        )

    async def push_audio_delta(
        self,
        encoded_pcm: np.ndarray,
        response_id: str,
        item_id: str,
        content_index: int,
        audio_queue: asyncio.Queue,
    ) -> None:
        if encoded_pcm.size == 0:
            return

        key = (response_id, item_id, content_index)
        with self._playback_lock:
            item_changed = self._receiving is None or self._receiving.key != key
            if item_changed:
                self._receiving = PlaybackState(
                    response_id,
                    item_id,
                    content_index,
                    self._playback_generation,
                )
            playback = self._receiving
            playback.received_samples += encoded_pcm.size

        if item_changed:
            self._pending_pcm = np.zeros(0, dtype=np.float32)
        self._pending_pcm = np.concatenate([self._pending_pcm, encoded_pcm])
        while self._pending_pcm.shape[0] >= self.frame_samples:
            frame = self._pending_pcm[: self.frame_samples]
            self._pending_pcm = self._pending_pcm[self.frame_samples :]
            await audio_queue.put(
                AudioFrame(
                    frame.astype(np.float32, copy=False),
                    lambda samples, state=playback: self.record_played(state, samples),
                )
            )

    def record_played(self, playback: PlaybackState, samples: int) -> None:
        with self._playback_lock:
            if playback.generation != self._playback_generation:
                return
            self._playing = playback
            playback.played_samples += samples

    async def handle_interruption(
        self,
        ws: websockets.ClientConnection | None,
        audio_queue: asyncio.Queue,
        interrupt_response: bool,
    ) -> None:
        if not interrupt_response:
            return
        with self._playback_lock:
            has_active_output = bool(
                self._active_response_ids
                or self._receiving is not None
                or self._playing is not None
            )
        if not has_active_output and audio_queue.empty():
            return

        self._pending_pcm = np.zeros(0, dtype=np.float32)
        dropped_frames = 0
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
                dropped_frames += 1
            except asyncio.QueueEmpty:
                break
        audio_queue.put_nowait(AudioInterrupted())

        with self._playback_lock:
            playback = self._playing
            receiving = self._receiving
            self._receiving = None
            self._playing = None
            self._playback_generation += 1
            cancelled = set(self._active_response_ids)
            self._cancelled_response_ids.update(cancelled)

        if receiving is not None:
            received_ms = int(
                receiving.received_samples / self.output_sample_rate * 1000
            )
            played_ms = int(
                min(receiving.played_samples, receiving.received_samples)
                / self.output_sample_rate
                * 1000
            )
            print(
                "[INTERRUPT] local playback cleared: "
                f"dropped_queue_frames={dropped_frames}, "
                f"received_ms={received_ms}, played_ms={played_ms}, "
                f"cancelled_responses={sorted(cancelled) or 'none'}",
                flush=True,
            )

        if playback is None or ws is None:
            return

        played_samples = min(playback.played_samples, playback.received_samples)
        if played_samples > 0:
            truncate_ms = int(played_samples / self.output_sample_rate * 1000)
            print(
                "[INTERRUPT] sending conversation.item.truncate "
                f"(item_id={playback.item_id}, audio_end_ms={truncate_ms})",
                flush=True,
            )
            await ws.send(
                json.dumps(
                    {
                        "type": "conversation.item.truncate",
                        "item_id": playback.item_id,
                        "content_index": playback.content_index,
                        "audio_end_ms": truncate_ms,
                    }
                )
            )
