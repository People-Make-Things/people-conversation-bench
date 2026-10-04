"""EOT response-similarity scoring for rollout traces."""

from __future__ import annotations

import re
import threading
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from bench.trace import SCORED, RolloutTrace

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embedding_model = None
_embedding_lock = threading.Lock()


@dataclass(frozen=True)
class ResponseSimilarityScore:
    eval: ClassVar[str] = "response-similarity"
    status: str
    token_f1: float
    embedding_cosine: float
    human_text: str
    model_text: str
    human_word_count: int
    model_word_count: int
    empty_response: bool
    timed_out: bool
    embedding_model: str

    def to_dict(self) -> dict:
        return {
            "eval": self.eval,
            "status": self.status,
            "token_f1": self.token_f1,
            "embedding_cosine": self.embedding_cosine,
            "human_text": self.human_text,
            "model_text": self.model_text,
            "human_word_count": self.human_word_count,
            "model_word_count": self.model_word_count,
            "empty_response": self.empty_response,
            "timed_out": self.timed_out,
            "embedding_model": self.embedding_model,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ResponseSimilarityScore:
        return cls(
            status=str(data.get("status", SCORED)),
            token_f1=float(data["token_f1"]),
            embedding_cosine=float(data["embedding_cosine"]),
            human_text=str(data.get("human_text", "")),
            model_text=str(data.get("model_text", "")),
            human_word_count=int(data.get("human_word_count", 0)),
            model_word_count=int(data.get("model_word_count", 0)),
            empty_response=bool(data.get("empty_response", False)),
            timed_out=bool(data.get("timed_out", False)),
            embedding_model=str(data.get("embedding_model", EMBEDDING_MODEL)),
        )


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(human_text: str, model_text: str) -> float:
    human = Counter(tokenize(human_text))
    model = Counter(tokenize(model_text))
    if not human or not model:
        return 0.0
    overlap = sum((human & model).values())
    precision = overlap / sum(model.values())
    recall = overlap / sum(human.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    norm_left = float(np.linalg.norm(left))
    norm_right = float(np.linalg.norm(right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    return float(np.dot(left, right) / (norm_left * norm_right))


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    # Late import: report/summarize only needs ResponseSimilarityScore, not
    # sentence-transformers / torch.
    from sentence_transformers import SentenceTransformer

    global _embedding_model
    with _embedding_lock:
        if _embedding_model is None:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        vectors = _embedding_model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    return np.asarray(vectors, dtype=np.float32)


def embedding_cosine(
    human_text: str,
    model_text: str,
    embed: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> float:
    if not human_text.strip() or not model_text.strip():
        return 0.0
    encode = embed or embed_texts
    vectors = np.asarray(encode([human_text, model_text]), dtype=np.float32)
    if vectors.shape[0] < 2:
        return 0.0
    return cosine_similarity(vectors[0], vectors[1])


def score_response_similarity(
    trace: RolloutTrace,
    *,
    human_text: str,
    model_text: str,
    embed: Callable[[Sequence[str]], np.ndarray] | None = None,
) -> ResponseSimilarityScore:
    if trace.expected_action != "eot":
        raise ValueError(
            f"response-similarity scoring requires eot, got {trace.expected_action}"
        )
    return ResponseSimilarityScore(
        status=trace.status,
        token_f1=token_f1(human_text, model_text),
        embedding_cosine=embedding_cosine(human_text, model_text, embed=embed),
        human_text=human_text,
        model_text=model_text,
        human_word_count=len(tokenize(human_text)),
        model_word_count=len(tokenize(model_text)),
        empty_response=not model_text.strip(),
        timed_out=trace.timed_out,
        embedding_model=EMBEDDING_MODEL,
    )
