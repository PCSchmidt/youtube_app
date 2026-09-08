"""Minimal FastAPI serving layer (Stage 1 scope: one chat endpoint).

POST /chat {url|file, question} runs ingest -> chunk -> embed -> index ->
retrieve -> generate and returns the grounded answer plus what was retrieved.
Injectors make the embedder/provider swappable so tests stay offline.

Stage 5 adds additive serving observability: one structured JSON log line per
request (stdout) and GET /metrics backed by in-process counters. Contracts of
/health and /chat are unchanged.
"""

from __future__ import annotations

import time

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from yt_rag.errors import IngestError
from yt_rag.observability import (
    PROM_ERROR_CLASSES,
    Metrics,
    PrometheusMetrics,
    get_logger,
    log_request,
    new_request_id,
)
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
    pipeline_factory=None,
) -> FastAPI:
    """Build the app. Defaults use the pinned real model path at runtime; the
    test suite injects a HashEmbedder + StubProvider to stay offline.

    pipeline_factory (optional, test-only override): returns a fully custom
    RAGPipeline per request, e.g. one whose retriever returns nothing to
    exercise the empty-result proxy. The production path ignores it.
    """
    metrics = Metrics()
    prom = PrometheusMetrics()
    logger = get_logger()

    def new_pipeline() -> RAGPipeline:
        if pipeline_factory is not None:
            return pipeline_factory()
        return RAGPipeline(embedder=embedder, provider=provider, top_k=top_k)

    app = FastAPI(title="yt_rag", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        started = time.perf_counter()
        payload = {"status": "ok"}
        prom.observe_request(
            endpoint="/health", method="GET", status=200, duration_s=time.perf_counter() - started
        )
        log_request(
            logger,
            request_id=new_request_id(),
            endpoint="/health",
            status=200,
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return payload

    @app.get("/metrics")
    def metrics_endpoint() -> dict:
        """In-process counters (reset on restart); /metrics does not count itself."""
        started = time.perf_counter()
        payload = metrics.snapshot()
        log_request(
            logger,
            request_id=new_request_id(),
            endpoint="/metrics",
            status=200,
            total_ms=round((time.perf_counter() - started) * 1000, 2),
        )
        return payload

    @app.get(
        "/metrics/prometheus",
        response_class=Response,
        responses={200: {"content": {"text/plain; version=0.0.4; charset=utf-8": {}}}},
    )
    def metrics_prometheus() -> Response:
        """Prometheus text exposition of the generic families.

        Rendered before recording itself, so this scrape does not appear in
        its own output (same convention as /metrics)."""
        started = time.perf_counter()
        body = prom.render()
        prom.observe_request(
            endpoint="/metrics/prometheus",
            method="GET",
            status=200,
            duration_s=time.perf_counter() - started,
        )
        return Response(
            content=body,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        request_id = new_request_id()
        pipeline = new_pipeline()
        started = time.perf_counter()
        status = 200
        error_class = None
        result = None
        try:
            if req.file:
                pipeline.ingest_from_file(req.file)
            elif req.url:
                pipeline.ingest_from_url(req.url)
            else:
                raise IngestError("provide either 'url' or 'file'")
            result = pipeline.ask(req.question)
            return ChatResponse(**result)
        except IngestError as exc:
            status, error_class = 422, "IngestError"
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (RuntimeError, ValueError) as exc:
            status, error_class = 400, type(exc).__name__
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # keep FastAPI's 500 behavior for the unexpected
            status, error_class = 500, type(exc).__name__
            raise
        finally:
            # One structured JSON line per request. Never logs the question,
            # the answer, retrieved text, or any API key: counts/scores/ids only.
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            prom.observe_request(
                endpoint="/chat",
                method="POST",
                status=status,
                duration_s=total_ms / 1000,
            )
            # Keep Prometheus labels bounded: unexpected exception classes
            # collapse to "Unhandled" (same defensive fallback as the log line).
            prom_error_class = error_class or "Unhandled"
            if prom_error_class not in PROM_ERROR_CLASSES:
                prom_error_class = "Unhandled"
            if error_class is not None:
                prom.observe_error(endpoint="/chat", method="POST", error_class=prom_error_class)
            timings = getattr(pipeline, "last_timings", None) or {}
            retrieval_s = timings.get("retrieval_s")
            generation_s = timings.get("generation_s")
            retrieval_ms = round(retrieval_s * 1000, 2) if retrieval_s is not None else None
            generation_ms = round(generation_s * 1000, 2) if generation_s is not None else None
            if result is not None:
                retrieved = result.get("retrieved", [])
                retrieved_count = len(retrieved)
                top_score = retrieved[0]["score"] if retrieved else None
                metrics.record_success(
                    total_ms=total_ms,
                    retrieval_ms=retrieval_ms if retrieval_ms is not None else 0.0,
                    generation_ms=generation_ms if generation_ms is not None else 0.0,
                    retrieved_count=retrieved_count,
                    top_score=top_score,
                )
                log_request(
                    logger,
                    request_id=request_id,
                    endpoint="/chat",
                    status=status,
                    total_ms=total_ms,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                    error_class=error_class,
                    question_chars=len(req.question),
                    retrieved_count=retrieved_count,
                    top_score=round(top_score, 4) if top_score is not None else None,
                    empty_result=retrieved_count == 0,
                )
            else:
                metrics.record_error(total_ms=total_ms, error_class=error_class or "Unhandled")
                log_request(
                    logger,
                    request_id=request_id,
                    endpoint="/chat",
                    status=status,
                    total_ms=total_ms,
                    retrieval_ms=None,
                    generation_ms=None,
                    error_class=error_class or "Unhandled",
                    question_chars=len(req.question),
                    retrieved_count=None,
                    top_score=None,
                    empty_result=None,
                )

    return app


app = create_app()
