"""Measure the teacher-forced text stream this repo builds against upstream's.

Answers two questions about `text_alignment.align_words_to_frames` that cannot
be answered from a stub tokenizer:

1. How much text the real point histories actually carry, i.e. whether the
   forced stream is so close to all-PAD that it is out of distribution.
2. How far the repo's word-queueing policy diverges from the policy kyutai's
   moshi-finetune interleaver uses by default (`keep_and_shift=False`, which
   drops the unfinished remainder of a word when the next one starts) --
   see moshi-finetune finetune/data/interleaver.py build_token_stream.

Needs the real sentencepiece model, which is not a repo dependency:

    uv venv /tmp/spm-venv && uv pip install --python /tmp/spm-venv/bin/python \
        sentencepiece==0.2 sphn numpy
    /tmp/spm-venv/bin/python data/analysis/text_conditioning/alignment_audit.py \
        --tokenizer /path/to/tokenizer_spm_32k_3.model
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import sentencepiece
import sphn

FRAME_SAMPLES = 1920
SAMPLE_RATE = 24000
FRAME_RATE = SAMPLE_RATE / FRAME_SAMPLES
PAD_TOKEN = 3
EPAD_TOKEN = 0


def tokenize(words: list[dict], tokenizer) -> list[tuple[int, list[int]]]:
    tokenized = []
    for word in words:
        pieces = [int(piece) for piece in tokenizer.encode(word["text"])]
        if pieces:
            tokenized.append((max(0, int(word["start_sec"] * FRAME_RATE)), pieces))
    return tokenized


def lay_out(tokenized, total_frames: int, keep_and_shift: bool):
    """Return the stream, the frame each word landed on, and pieces lost."""
    aligned = [PAD_TOKEN] * total_frames
    pending: deque = deque()
    placed: dict[int, int] = {}
    next_word = 0
    dropped = 0
    for frame_idx in range(total_frames):
        while next_word < len(tokenized) and tokenized[next_word][0] <= frame_idx:
            entry = [(next_word, piece) for piece in tokenized[next_word][1]]
            if keep_and_shift:
                pending.extend(entry)
            else:
                dropped += len(pending)
                pending = deque(entry)
            next_word += 1
        if not pending:
            continue
        if frame_idx > 0 and aligned[frame_idx - 1] == PAD_TOKEN:
            aligned[frame_idx - 1] = EPAD_TOKEN
        word_index, piece = pending.popleft()
        aligned[frame_idx] = piece
        placed.setdefault(word_index, frame_idx)
    return aligned, placed, dropped + len(pending)


def audit_point(point_dir: Path, tokenizer) -> dict | None:
    meta = json.loads((point_dir / "point.json").read_text())
    rollout = meta["rollout"]
    transcript = point_dir / rollout["assistant_transcript"]
    words = json.loads(transcript.read_text()).get("words", [])
    pcm, rate = sphn.read(str(point_dir / rollout["assistant_audio"]))
    total_frames = pcm.shape[-1] // FRAME_SAMPLES
    if not words or not total_frames:
        return None

    tokenized = tokenize(words, tokenizer)
    repo, repo_placed, repo_lost = lay_out(tokenized, total_frames, keep_and_shift=True)
    upstream, up_placed, up_lost = lay_out(tokenized, total_frames, keep_and_shift=False)
    drift = [
        (repo_placed[index] - tokenized[index][0]) / FRAME_RATE for index in repo_placed
    ]
    spoken = sum(1 for token in repo if token not in (PAD_TOKEN, EPAD_TOKEN))
    return {
        "point": point_dir.name,
        "history_sec": round(total_frames / FRAME_RATE, 2),
        "frames": total_frames,
        "words": len(tokenized),
        "pieces": sum(len(pieces) for _, pieces in tokenized),
        "spoken_frames": spoken,
        "pad_fraction": round(1 - spoken / total_frames, 3),
        "epad_frames": repo.count(EPAD_TOKEN),
        "max_word_drift_sec": round(max(drift), 3),
        "mean_word_drift_sec": round(sum(drift) / len(drift), 4),
        "pieces_lost_repo": repo_lost,
        "pieces_lost_upstream": up_lost,
        "streams_agree": repo == upstream,
        "frames_differing": sum(1 for a, b in zip(repo, upstream) if a != b),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--points", type=Path, default=Path("data/processed"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/analysis/text_conditioning/alignment_audit.json"),
    )
    args = parser.parse_args()

    tokenizer = sentencepiece.SentencePieceProcessor(str(args.tokenizer))
    rows = []
    for point_dir in sorted(args.points.glob("*/points/*")):
        row = audit_point(point_dir, tokenizer)
        if row is not None:
            rows.append(row)

    header = f'{"point":9}{"hist_s":>8}{"words":>7}{"pieces":>7}{"PAD%":>7}{"maxdrift":>9}{"lostUp":>7}{"diffFrames":>11}'
    print(header)
    for row in rows:
        print(
            f'{row["point"]:9}{row["history_sec"]:8.1f}{row["words"]:7d}'
            f'{row["pieces"]:7d}{row["pad_fraction"] * 100:7.1f}'
            f'{row["max_word_drift_sec"]:9.2f}{row["pieces_lost_upstream"]:7d}'
            f'{row["frames_differing"]:11d}'
        )

    summary = {
        "points": len(rows),
        "pad_fraction_min": min(row["pad_fraction"] for row in rows),
        "pad_fraction_max": max(row["pad_fraction"] for row in rows),
        "worst_word_drift_sec": max(row["max_word_drift_sec"] for row in rows),
        "pieces_lost_repo": sum(row["pieces_lost_repo"] for row in rows),
        "pieces_lost_upstream": sum(row["pieces_lost_upstream"] for row in rows),
        "total_pieces": sum(row["pieces"] for row in rows),
        "points_where_policies_agree": sum(1 for row in rows if row["streams_agree"]),
        "worst_frames_differing": max(row["frames_differing"] for row in rows),
    }
    print("\n" + json.dumps(summary, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "points": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
