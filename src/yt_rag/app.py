"""Minimal FastAPI serving layer (Stage 1 scope: one chat endpoint).

POST /chat {url|file, question} runs ingest -> chunk -> embed -> index ->
retrieve -> generate and returns the grounded answer plus what was retrieved.
Injectors make the embedder/provider swappable so tests stay offline.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from yt_rag.errors import IngestError
from yt_rag.pipeline import RAGPipeline


class ChatRequest(BaseModel):
    url: str | None = Field(default=None, description="YouTube URL or video ID")
    file: str | None = Field(default=None, description="Path to a cached transcript .txt")
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    retrieved: list[dict]


def create_app(
    embedder=None,
    provider=None,
    top_k: int = 4,
) -> FastAPI:
    """Build the app. Defaults use the pinned real model path at runtime; the
    test suite injects a HashEmbedder + StubProvider to stay offline."""

    def new_pipeline() -> RAGPipeline:
        return RAGPipeline(embedder=embedder, provider=provider, top_k=top_k)

    app = FastAPI(title="yt_rag", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        pipeline = new_pipeline()
        try:
            if req.file:
                pipeline.ingest_from_file(req.file)
            elif req.url:
                pipeline.ingest_from_url(req.url)
            else:
                raise IngestError("provide either 'url' or 'file'")
            result = pipeline.ask(req.question)
        except IngestError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ChatResponse(**result)

    return app


app = create_app()
