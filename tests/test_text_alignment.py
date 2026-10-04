"""Tests for the teacher-forced PersonaPlex text stream.

The stream is PAD everywhere except a run of one sentencepiece token per frame
per word, preceded by a single EPAD. `first_two_pieces_per_word` reproduces the
previous implementation so these tests can show they catch its losses.
"""

from __future__ import annotations

from dataclasses import dataclass

from bench.audio import MODEL_FRAME_SAMPLES
from text_alignment import (
    EPAD_TOKEN,
    PAD_TOKEN,
    align_words_to_frames,
    frame_count,
    frame_rate_hz,
)


@dataclass(frozen=True)
class Word:
    text: str
    start_sec: float


class CharacterTokenizer:
    """One token per character, so every word is multi-piece."""

    def encode(self, text: str) -> list[int]:
        return [ord(character) for character in text]


def align(words: list[Word], frames: int) -> list[int]:
    return align_words_to_frames(
        words,
        MODEL_FRAME_SAMPLES,
        frames * MODEL_FRAME_SAMPLES,
        CharacterTokenizer(),
    )


def first_two_pieces_per_word(words: list[Word], frames: int) -> list[int]:
    """Pre-fix behaviour: two tokens per word, later words overwrite earlier ones."""
    aligned = [PAD_TOKEN] * frames
    tokenizer = CharacterTokenizer()
    for word in words:
        index = int(word.start_sec * frame_rate_hz(MODEL_FRAME_SAMPLES))
        pieces = tokenizer.encode(word.text)
        if index >= frames or not pieces:
            continue
        aligned[index] = pieces[0]
        if len(pieces) > 1 and index + 1 < frames:
            aligned[index + 1] = pieces[1]
    return aligned


def spoken_tokens(aligned: list[int]) -> list[int]:
    return [token for token in aligned if token not in (PAD_TOKEN, EPAD_TOKEN)]


def test_empty_history_is_all_padding() -> None:
    assert align([], 6) == [PAD_TOKEN] * 6


def test_word_occupies_one_frame_per_token_after_an_epad() -> None:
    aligned = align([Word("hi", 0.4)], 10)
    assert aligned[4] == EPAD_TOKEN
    assert aligned[5:7] == [ord("h"), ord("i")]
    assert aligned[:4] == [PAD_TOKEN] * 4
    assert aligned[7:] == [PAD_TOKEN] * 3


def test_word_at_the_first_frame_has_nowhere_to_put_an_epad() -> None:
    aligned = align([Word("ab", 0.0)], 4)
    assert aligned == [ord("a"), ord("b"), PAD_TOKEN, PAD_TOKEN]


def test_words_sharing_a_frame_queue_instead_of_overwriting() -> None:
    words = [Word("ab", 0.08), Word("cd", 0.08)]
    aligned = align(words, 8)
    assert spoken_tokens(aligned) == [ord(character) for character in "abcd"]
    assert aligned[0] == EPAD_TOKEN
    assert spoken_tokens(first_two_pieces_per_word(words, 8)) != spoken_tokens(aligned)


def test_long_words_spill_forward_without_dropping_the_next_word() -> None:
    words = [Word("abcdefgh", 0.0), Word("xy", 0.16)]
    aligned = align(words, 16)
    assert spoken_tokens(aligned) == [ord(character) for character in "abcdefghxy"]


def test_every_token_of_every_word_survives() -> None:
    words = [Word("hello", 0.0), Word("there", 0.8), Word("friend", 1.6)]
    aligned = align(words, 40)
    assert spoken_tokens(aligned) == [
        ord(character) for character in "hellotherefriend"
    ]
    # The pre-fix version kept only the first two pieces of each word.
    assert len(spoken_tokens(first_two_pieces_per_word(words, 40))) == 6


def test_epad_marks_only_the_frame_before_each_run() -> None:
    aligned = align([Word("hello", 0.0), Word("there", 0.8)], 40)
    assert aligned.count(EPAD_TOKEN) == 1
    assert aligned[9] == EPAD_TOKEN


def test_words_beyond_the_history_are_ignored() -> None:
    aligned = align([Word("late", 99.0)], 5)
    assert aligned == [PAD_TOKEN] * 5


def test_frame_count_requires_whole_frames() -> None:
    assert frame_count(MODEL_FRAME_SAMPLES * 3, MODEL_FRAME_SAMPLES) == 3
