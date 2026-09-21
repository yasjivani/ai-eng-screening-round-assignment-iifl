"""BM25 retrieval over the generated policy chunk index."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path(__file__).resolve().parents[1] / "data" / "chunks.json"

SYNONYMS = {
    "prepay": ["foreclosure", "prepayment"],
    "prepayment": ["foreclosure"],
    "preclose": ["foreclosure"],
    "close": ["foreclosure"],
    "early": ["foreclosure", "prepayment"],
    "penalty": ["penal", "charges"],
    "fine": ["penal", "charges"],
    "late": ["penal", "overdue"],
    "roi": ["interest", "rate"],
    "emi": ["instalment", "interest"],
    "bounce": ["nach", "mandate"],
    "colending": ["co-lending", "lending", "partner"],
    "gst": ["tax", "goods", "services"],
}

_TOKEN_RE = re.compile(r"[a-z0-9₹%.]+")

STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "have", "how", "i", "if", "in", "is", "it", "many", "me",
    "much", "my", "of", "on", "or", "our", "shall", "should", "so", "that",
    "the", "their", "there", "these", "this", "to", "was", "we", "what", "when",
    "where", "which", "who", "will", "with", "would", "you", "your",
}


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [
        t.strip(".")
        for t in tokens
        if t.strip(".") and t.strip(".") not in STOPWORDS
    ]

    expanded = list(tokens)
    for token in tokens:
        expanded.extend(SYNONYMS.get(token, []))
    return expanded


def index_tokens(chunk: dict) -> list[str]:
    head = f"{chunk.get('section', '')} {chunk['text'][:120]}"
    return tokenize(head) * 3 + tokenize(chunk["text"])


@dataclass
class Hit:
    chunk: dict
    score: float


class Retriever:
    def __init__(self, chunks: list[dict]):
        if not chunks:
            raise ValueError("Chunk index is empty. Run `python -m src.ingest` first.")
        self.chunks = chunks
        self._bm25 = BM25Okapi([index_tokens(c) for c in chunks])

    @classmethod
    def from_file(cls, path: Path = CHUNKS_PATH) -> "Retriever":
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m src.ingest` first."
            )
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def search(self, query: str, k: int = 4) -> list[Hit]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]

        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]))
            for i in ranked
            if scores[i] > 0
        ]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return Retriever.from_file()
