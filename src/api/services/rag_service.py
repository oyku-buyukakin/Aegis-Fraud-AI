from __future__ import annotations

import json
import math
import re
from pathlib import Path

from ..schemas import RAGChunk, RAGQueryRequest, RAGQueryResponse

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "of", "to", "for", "and", "or", "in", "on", "at", "by",
    "with", "from", "as", "that", "this", "these", "those", "it", "its", "what",
    "when", "where", "why", "how", "who", "which", "whom", "make", "makes", "made",
    "making", "happen", "happens", "happened", "i", "you", "we", "they", "he", "she",
    "my", "your", "our", "their", "me", "us", "them", "about", "into", "over", "under",
    "than", "then", "can", "could", "should", "would", "will", "shall", "may", "might",
    "if", "so", "such", "also", "there", "here", "out", "up", "down", "not", "no",
}


class RagService:

    def __init__(self, kb_dir: str, min_score: float = 0.8, max_chars: int = 1500) -> None:
        self._kb_dir = Path(kb_dir)
        self._min_score = min_score
        self._max_chars = max_chars
        self._chunks: list[dict] | None = None

    def _load(self) -> list[dict]:
        if self._chunks is not None:
            return self._chunks
        chunks: list[dict] = []
        if self._kb_dir.exists():
            for path in sorted(self._kb_dir.iterdir()):
                if path.suffix == ".md":
                    text = path.read_text(encoding="utf-8").strip()
                    for i, chunk in enumerate(re.split(r"\n(?=#{1,3} )|(?<=\n)---+\n", text)):
                        chunk = chunk.strip()
                        if chunk:
                            chunks.append({"source": path.name, "chunk_id": i, "text": chunk})
                elif path.suffix == ".json":
                    data = json.loads(path.read_text(encoding="utf-8"))
                    items = data if isinstance(data, list) else [data]
                    for item in items:
                        chunks.append({"source": path.name, "chunk_id": len(chunks), "text": json.dumps(item)})
        self._chunks = chunks
        return self._chunks

    def chunk_count(self) -> int:
        return len(self._load())

    @staticmethod
    def _is_readable(chunk: dict) -> bool:
        if chunk["source"].endswith(".json"):
            return False
        head = chunk["text"].lstrip().lower()
        if head.startswith(("### synthetic case", "## synthetic case", "{")):
            return False
        return True

    def _score(self, query: str, text: str) -> float:
        qterms = {t for t in re.findall(r"[a-z0-9_]+", query.lower()) if t not in _STOPWORDS}
        tterms = re.findall(r"[a-z0-9_]+", text.lower())
        if not qterms or not tterms:
            return 0.0
        freq = sum(1 for t in tterms if t in qterms)
        return round(freq / (1 + math.log(1 + len(tterms))), 4)

    def search(self, req: RAGQueryRequest) -> RAGQueryResponse:
        scored = sorted(
            ({**c, "score": self._score(req.query, c["text"])} for c in self._load() if self._is_readable(c)),
            key=lambda x: x["score"],
            reverse=True,
        )
        top = [c for c in scored[: req.top_k] if c["score"] >= self._min_score]
        return RAGQueryResponse(
            query=req.query,
            results=[RAGChunk(source=c["source"], score=c["score"], text=c["text"][: self._max_chars]) for c in top],
        )