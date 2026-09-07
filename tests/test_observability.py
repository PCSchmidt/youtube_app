"""Stage 5 observability tests: /metrics counters, structured JSON logs, proxy.

Offline and docker-free: HashEmbedder + StubProvider (or a test-only pipeline
factory that forces an empty retrieval). Fast additions only.
"""

import json
import logging

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from yt_rag.app import create_app
from yt_rag.embeddings import HashEmbedder
from yt_rag.generation import StubProvider
from yt_rag.observability import LOGGER_NAME
from yt_rag.pipeline import RAGPipeline


class ListHandler(logging.Handler):
    """Collects log messages in memory so tests can parse the JSON lines."""

    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


@pytest.fixture(scope="module")
def client(teal_fixture):
    app = create_app(embedder=HashEmbedder(dim=384), provider=StubProvider())
    with TestClient(app) as c:
        c.teal_fixture = str(teal_fixture)
        yield c


def _chat(client, question="how to optimize a linkedin profile"):
    return client.post(
        "/chat",
        json={"file": client.teal_fixture, "question": question},
    )


def test_metrics_increment_after_chat(client):
    before = client.get("/metrics").json()
    assert _chat(client).status_code == 200
    after = client.get("/metrics").json()
    assert after["request_count"] == before["request_count"] + 1
    assert after["error_count"] == before["error_count"]
    assert after["latency_ms"]["total"]["count"] == before["latency_ms"]["total"]["count"] + 1
    assert (
        after["latency_ms"]["retrieval"]["count"] == before["latency_ms"]["retrieval"]["count"] + 1
    )
    assert (
        after["latency_ms"]["generation"]["count"]
        == before["latency_ms"]["generation"]["count"] + 1
    )
    assert after["latency_ms"]["total"]["max_ms"] >= 0.0
    assert after["latency_ms"]["retrieval"]["p50_ms"] >= 0.0


def test_metrics_errors_and_error_rate(client):
    before = client.get("/metrics").json()
    assert client.post("/chat", json={"question": "hi"}).status_code == 422
    after = client.get("/metrics").json()
    assert after["error_count"] == before["error_count"] + 1
    assert (
        after["error_classes"].get("IngestError", 0)
        == before["error_classes"].get("IngestError", 0) + 1
    )
    assert 0.0 < after["error_rate"] <= 1.0


def test_empty_result_proxy_counted(teal_fixture):
    """Force retrieved_count == 0 via a test-only pipeline factory."""

    class EmptyRetriever:
        def retrieve(self, query: str):
            return []

    class NullProvider:
        def generate(self, question: str, retrieved) -> str:
            return "no retrieved chunks"

    class EmptyResultPipeline(RAGPipeline):
        def ingest_transcript(self, transcript):
            chunks = super().ingest_transcript(transcript)
            # The /chat handler always ingests per request; keep the empty
            # retriever so retrieval yields zero chunks for this test.
            self.retriever = EmptyRetriever()
            return chunks

    def factory() -> RAGPipeline:
        return EmptyResultPipeline(embedder=HashEmbedder(dim=384), provider=NullProvider())

    app = create_app(pipeline_factory=factory)
    with TestClient(app) as c:
        r = c.post("/chat", json={"file": str(teal_fixture), "question": "anything"})
        assert r.status_code == 200
        assert r.json()["retrieved"] == []
        m = c.get("/metrics").json()
        proxy = m["retrieval_quality_proxy"]
        assert proxy["empty_result_count"] == 1
        assert proxy["empty_result_rate"] == 1.0
        assert proxy["mean_top_score"] is None


def test_chat_log_is_structured_json(client):
    logger = logging.getLogger(LOGGER_NAME)
    handler = ListHandler()
    logger.addHandler(handler)
    try:
        assert _chat(client).status_code == 200
        assert client.post("/chat", json={"question": "hi"}).status_code == 422
    finally:
        logger.removeHandler(handler)

    records = [json.loads(m) for m in handler.messages]
    chat_ok = [r for r in records if r.get("endpoint") == "/chat" and r.get("status") == 200]
    assert chat_ok, records
    rec = chat_ok[-1]
    for field in (
        "request_id",
        "endpoint",
        "status",
        "total_ms",
        "retrieval_ms",
        "generation_ms",
        "error_class",
        "retrieved_count",
        "top_score",
        "empty_result",
    ):
        assert field in rec, (field, rec)
    assert rec["error_class"] is None
    assert rec["total_ms"] >= 0
    assert rec["retrieval_ms"] >= 0
    assert rec["generation_ms"] >= 0
    assert rec["retrieved_count"] > 0
    assert isinstance(rec["top_score"], float)
    assert rec["empty_result"] is False

    chat_err = [r for r in records if r.get("endpoint") == "/chat" and r.get("status") == 422]
    assert chat_err, records
    assert chat_err[-1]["error_class"] == "IngestError"


def test_logs_never_contain_prompt_or_retrieved_text(client):
    """No prompts/answers/retrieved text in logs: counts and scores only."""
    logger = logging.getLogger(LOGGER_NAME)
    handler = ListHandler()
    logger.addHandler(handler)
    try:
        # "profile" and "linkedin" appear in both the question and the
        # fixture transcript chunks; neither may leak into any log line.
        assert _chat(client, "how to optimize a linkedin profile").status_code == 200
    finally:
        logger.removeHandler(handler)
    assert handler.messages
    for message in handler.messages:
        assert "linkedin" not in message.lower()
        assert "profile" not in message.lower()
        # Every line must parse as a single JSON object (no free-form text).
        json.loads(message)
