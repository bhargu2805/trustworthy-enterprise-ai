from __future__ import annotations

import os
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from compliance.engine import ComplianceEngine, CheckOutcome
from retriever.retriever import retrieve
from retriever.ingest import ingest_texts

import ollama as ollama_client

app = FastAPI(title="Trustworthy Enterprise AI – Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

POLICY_PATH = os.getenv("POLICY_PATH", "./policies/policy.yaml")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
compliance = ComplianceEngine(POLICY_PATH)


class AskRequest(BaseModel):
    query: str
    tenant_id: str | None = "demo"
    top_k: int = 5


class AskResponse(BaseModel):
    answer: str | None
    sources: list[dict] = []
    blocked: bool = False
    flags: list[str] = []
    redactions: list[str] = []
    latency_ms: int = 0
    costs: dict = {}


class IngestRequest(BaseModel):
    documents: list[dict]


def _build_prompt(query: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {c['title']}\n{c['text']}"
        for i, c in enumerate(chunks)
    )
    return f"""You are a helpful enterprise AI assistant. Answer the question using only the provided context.
If the context does not contain enough information, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL) -> str:
    try:
        response = ollama_client.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")


@app.get("/health")
def health():
    return {"status": "ok", "model": OLLAMA_MODEL}


@app.post("/v1/ingest")
def ingest(req: IngestRequest):
    t0 = time.time()
    result = ingest_texts(req.documents)
    result["latency_ms"] = int((time.time() - t0) * 1000)
    return result


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest):
    t0 = time.time()

    ctx = retrieve(req.query, tenant_id=req.tenant_id or "demo", k=req.top_k)
    chunks = ctx["chunks"]

    if not chunks:
        return AskResponse(
            answer="No relevant documents found. Please ingest some documents first.",
            sources=[],
            latency_ms=int((time.time() - t0) * 1000)
        )

    prompt = _build_prompt(req.query, chunks)
    answer = _call_ollama(prompt)

    return AskResponse(
        answer=answer,
        sources=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
        latency_ms=int((time.time() - t0) * 1000),
        costs={"usd": 0}
    )


@app.post("/v1/ask/compliant", response_model=AskResponse)
def ask_compliant(req: AskRequest):
    t0 = time.time()

    # Pre-compliance check on user query
    pre: CheckOutcome = compliance.check(req.query, phase="pre")
    if pre.block:
        return AskResponse(
            answer=None,
            blocked=True,
            flags=pre.flags,
            redactions=pre.redactions,
            latency_ms=int((time.time() - t0) * 1000)
        )

    ctx = retrieve(req.query, tenant_id=req.tenant_id or "demo", k=req.top_k)
    chunks = ctx["chunks"]

    if not chunks:
        return AskResponse(
            answer="No relevant documents found. Please ingest some documents first.",
            sources=[],
            flags=pre.flags,
            latency_ms=int((time.time() - t0) * 1000)
        )

    prompt = _build_prompt(req.query, chunks)
    answer = _call_ollama(prompt)

    # Post-compliance check on LLM answer
    post: CheckOutcome = compliance.check(answer, phase="post")
    if post.block:
        return AskResponse(
            answer=None,
            blocked=True,
            flags=pre.flags + post.flags,
            redactions=pre.redactions + post.redactions,
            latency_ms=int((time.time() - t0) * 1000)
        )

    return AskResponse(
        answer=answer,
        sources=[{k: v for k, v in c.items() if k != "text"} for c in chunks],
        flags=pre.flags + post.flags,
        redactions=pre.redactions + post.redactions,
        latency_ms=int((time.time() - t0) * 1000),
        costs={"usd": 0}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)