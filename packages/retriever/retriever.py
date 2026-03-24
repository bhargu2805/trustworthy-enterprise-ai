from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/index.faiss")
META_PATH = os.getenv("FAISS_META_PATH", "data/index.meta")


@dataclass
class Chunk:
    chunk_id: str
    title: str
    text: str
    url: str | None = None
    score: float = 0.0


class Retriever:
    def __init__(self, index_path: str = INDEX_PATH, meta_path: str = META_PATH):
        self.index_path = index_path
        self.meta_path = meta_path
        self._model: SentenceTransformer | None = None
        self._index: faiss.Index | None = None
        self._meta: List[Dict[str, Any]] = []

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(MODEL_NAME)
        return self._model

    def _load_index(self):
        if self._index is None and Path(self.index_path).exists():
            self._index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self._meta = pickle.load(f)

    def is_ready(self) -> bool:
        return Path(self.index_path).exists() and Path(self.meta_path).exists()

    def retrieve(self, query: str, k: int = 5) -> List[Chunk]:
        self._load_index()

        if self._index is None or len(self._meta) == 0:
            return []

        model = self._get_model()
        t0 = time.time()
        query_vec = model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self._index.search(query_vec, min(k, len(self._meta)))
        latency_ms = int((time.time() - t0) * 1000)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            m = self._meta[idx]
            results.append(Chunk(
                chunk_id=m.get("chunk_id", f"chunk-{idx}"),
                title=m.get("title", "Untitled"),
                text=m.get("text", ""),
                url=m.get("url"),
                score=float(score)
            ))
        return results


# Module-level singleton
_retriever = Retriever()


def retrieve(query: str, tenant_id: str = "demo", k: int = 5) -> dict:
    chunks = _retriever.retrieve(query, k=k)
    return {
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "title": c.title,
                "text": c.text,
                "url": c.url,
                "score": round(c.score, 4)
            }
            for c in chunks
        ],
        "index_ready": _retriever.is_ready()
    }