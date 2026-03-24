from __future__ import annotations

import os
import pickle
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/index.faiss")
META_PATH = os.getenv("FAISS_META_PATH", "data/index.meta")
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def _clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def ingest_texts(documents: List[Dict[str, Any]], index_path: str = INDEX_PATH, meta_path: str = META_PATH) -> dict:
    """
    Ingest a list of documents into the FAISS index.

    Each document should have:
      - title: str
      - text: str
      - url: str (optional)

    Returns a summary dict with chunk_count and latency_ms.
    """
    import time
    t0 = time.time()

    Path(index_path).parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(MODEL_NAME)

    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        title = doc.get("title", "Untitled")
        url = doc.get("url")
        text = _clean_text(doc.get("text", ""))
        chunks = _chunk_text(text)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "title": title,
                "text": chunk_text,
                "url": url,
                "doc_chunk_index": i
            })

    if not all_chunks:
        return {"chunk_count": 0, "latency_ms": 0}

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=True).astype("float32")

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(all_chunks, f)

    latency_ms = int((time.time() - t0) * 1000)
    return {
        "chunk_count": len(all_chunks),
        "doc_count": len(documents),
        "latency_ms": latency_ms
    }


def ingest_from_file(file_path: str, title: str | None = None) -> dict:
    """Ingest a plain text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    text = path.read_text(encoding="utf-8")
    doc_title = title or path.stem
    return ingest_texts([{"title": doc_title, "text": text, "url": None}])


if __name__ == "__main__":
    sample_docs = [
        {
            "title": "Introduction to RAG",
            "text": """Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with language generation.
            Instead of relying solely on a language model's parametric knowledge, RAG retrieves relevant documents from an external
            knowledge base and uses them as context for generating answers. This approach improves factual accuracy and allows the
            system to work with up-to-date or domain-specific information that was not present during training.
            The retrieval component typically uses dense vector search with embeddings, while the generation component uses a
            large language model to synthesize an answer from the retrieved context.""",
            "url": None
        },
        {
            "title": "FAISS Vector Search",
            "text": """FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors.
            It is designed to handle large collections of vectors and supports both exact and approximate nearest neighbor search.
            In RAG systems, FAISS is used to store document embeddings and quickly find the most relevant chunks for a given query.
            The IndexFlatIP index performs exact inner product search, which is equivalent to cosine similarity when vectors are
            normalized. FAISS can search millions of vectors in milliseconds, making it suitable for production RAG systems.""",
            "url": None
        },
        {
            "title": "Enterprise AI Compliance",
            "text": """Enterprise AI systems must handle sensitive information responsibly. Key compliance requirements include
            PII (Personally Identifiable Information) detection to prevent leaking names, emails, and social security numbers.
            Secrets filtering ensures API keys, passwords, and tokens are not exposed in model outputs. Prompt injection prevention
            guards against malicious inputs designed to hijack the model's behavior. A policy-based compliance engine can enforce
            these rules using configurable YAML policies with regex patterns. Pre-processing checks validate user inputs before
            they reach the LLM, while post-processing checks validate the model's output before it is returned to the user.""",
            "url": None
        }
    ]

    print("Ingesting sample documents...")
    result = ingest_texts(sample_docs)
    print(f"Done: {result}")