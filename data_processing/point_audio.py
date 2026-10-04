"""Write shared point audio artifacts: duplex history, live input, EOT reference."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from bench.audio import pad_pcm_to_model_frames, write_wav_pcm16
from bench.points import ReferenceResponse
from bench.protocol import TimedWord
from bench.transcribe import transcribe_words
from data_processing.audio import read_wav_range
from data_processing.labels import RolloutWindow, SplitEvent, Turn
from utils.timed_words import write_timed_words

Transcribe = Callable[[Path], list[TimedWord]]


def transcript_for(
    channel_words: dict[str, list[dict]] | None,
    speaker: str,
    start_sec: float,
    end_sec: float,
    transcribe: Transcribe | None = None,
) -> Transcribe:
    if transcribe is not None:
        return transcribe
    if channel_words is not None:
        def from_channel(_path: Path) -> list[TimedWord]:
            return channel_timed_words(channel_words, speaker, start_sec, end_sec)

        return from_channel
    return transcribe_words


def channel_timed_words(
    channel_words: dict[str, list[dict]] | None,
    speaker: str,
    start_sec: float,
    end_sec: float,
) -> list[TimedWord]:
    words: list[TimedWord] = []
    for word in (channel_words or {}).get(speaker, []):
        word_start = float(word["start"])
        word_end = float(word["end"])
        if word_end <= start_sec or word_start >= end_sec:
            continue
        text = str(word.get("text") or "").strip()
        if not text:
            continue
        words.append(
            TimedWord(
                text=text,
                start_sec=max(word_start, start_sec) - start_sec,
                end_sec=min(word_end, end_sec) - start_sec,
            )
        )
    return words


def write_duplex_history(
    point_dir: Path,
    user_src: Path,
    assistant_src: Path,
    end_sec: float,
    sample_rate: int,
    transcribe: Transcribe = transcribe_words,
    start_sec: float = 0.0,
) -> None:
    user_pcm, _ = read_wav_range(user_src, start_sec, end_sec, target_rate=sample_rate)
    assistant_pcm, _ = read_wav_range(
        assistant_src, start_sec, end_sec, target_rate=sample_rate
    )
    write_wav_pcm16(
        point_dir / "rollout_user_context.wav",
        pad_pcm_to_model_frames(user_pcm),
        sample_rate,
    )
    write_wav_pcm16(
        point_dir / "rollout_assistant.wav",
        pad_pcm_to_model_frames(assistant_pcm),
        sample_rate,
    )
    write_timed_words(
        point_dir / "rollout_assistant_transcript.json",
        transcribe(point_dir / "rollout_assistant.wav"),
    )


def write_rollout_input_audio(
    point_dir: Path,
    speaker_path: Path,
    event: SplitEvent,
    window: RolloutWindow,
    sample_rate: int,
) -> str:
    if event.split_type == "pause_start":
        input_audio = "rollout_user_input_pause_start.wav"
        pause_start_pcm, _ = read_wav_range(
            speaker_path,
            window.turn_start,
            window.checkpoint,
        )
        pause_end_pcm, _ = read_wav_range(
            speaker_path,
            window.turn_start,
            window.window_end,
        )
        write_wav_pcm16(
            point_dir / input_audio,
            pad_pcm_to_model_frames(pause_start_pcm),
            sample_rate,
        )
        write_wav_pcm16(
            point_dir / "rollout_user_input_pause_end.wav",
            pad_pcm_to_model_frames(pause_end_pcm),
            sample_rate,
        )
        return input_audio
    input_audio = "rollout_user_input.wav"
    rollout_user_input, _ = read_wav_range(
        speaker_path,
        window.turn_start,
        window.input_end,
    )
    write_wav_pcm16(
        point_dir / input_audio,
        pad_pcm_to_model_frames(rollout_user_input),
        sample_rate,
    )
    return input_audio


def write_reference_response(
    point_dir: Path,
    speaker_path: Path,
    turn: Turn,
    sample_rate: int,
    transcribe: Transcribe = transcribe_words,
) -> ReferenceResponse:
    audio_name = "rollout_reference_response.wav"
    transcript_name = "rollout_reference_response_transcript.json"
    pcm, _ = read_wav_range(speaker_path, turn.start, turn.end)
    write_wav_pcm16(point_dir / audio_name, pcm, sample_rate)
    words = transcribe(point_dir / audio_name)
    write_timed_words(point_dir / transcript_name, words)
    return ReferenceResponse(
        duration_sec=turn.end - turn.start,
        word_count=len(words),
        audio=audio_name,
        transcript=transcript_name,
        start_sec=turn.start,
        end_sec=turn.end,
    )
