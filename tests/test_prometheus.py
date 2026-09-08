"""Phase 2 tests: GET /metrics/prometheus (stdlib text exposition writer).

Offline and deterministic: the writer is unit-tested against fixed state, the
endpoint is exercised via TestClient with injected offline stubs. The JSON
GET /metrics contract is covered by test_observability.py and must not change.
"""

import re

import pytest

fastapi = pytest.importorskip("fastapi")
TestClient = pytest.importorskip("fastapi.testclient").TestClient

from yt_rag.app import create_app
from yt_rag.embeddings import HashEmbedder
from yt_rag.generation import StubProvider
from yt_rag.observability import PROM_ENDPOINTS, PrometheusMetrics
from yt_rag.pipeline import RAGPipeline

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

METRIC_NAMES = (
    "yt_rag_requests_total",
    "yt_rag_errors_total",
    "yt_rag_request_latency_seconds_bucket",
    "yt_rag_request_latency_seconds_sum",
    "yt_rag_request_latency_seconds_count",
    "yt_rag_up",
)

ALLOWED_ERROR_CLASSES = {
    "IngestError",
    "GenerationError",
    "IndexStateError",
    "RuntimeError",
    "ValueError",
    "Unhandled",
}
ALLOWED_METHODS = {"GET", "POST"}
ALLOWED_STATUSES = {"200", "400", "404", "422", "500"}


@pytest.fixture(scope="module")
def client(teal_fixture):
    app = create_app(embedder=HashEmbedder(dim=384), provider=StubProvider())
    with TestClient(app) as c:
        c.teal_fixture = str(teal_fixture)
        yield c


@pytest.fixture()
def fresh_client(teal_fixture):
    """Function-scoped app: zero prior traffic, for exact-count assertions."""
    app = create_app(embedder=HashEmbedder(dim=384), provider=StubProvider())
    with TestClient(app) as c:
        c.teal_fixture = str(teal_fixture)
        yield c


def _get_prom(client):
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"] == CONTENT_TYPE
    return r.text


def _series(text: str, name: str) -> list[tuple[dict[str, str], int | float]]:
    """Parse sample lines for one family into (labels, value) pairs."""
    out = []
    for m in re.finditer(rf"^{re.escape(name)}(?:\{{([^}}]*)\}})? (\S+)$", text, re.MULTILINE):
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1) or ""))
        value = m.group(2)
        out.append((labels, float(value)))
    return out


def _lookup(text: str, name: str, labels: dict[str, str]) -> float | None:
    for found, value in _series(text, name):
        if found == labels:
            return value
    return None


def test_content_type_and_status(client):
    r = client.get("/metrics/prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"] == CONTENT_TYPE


def test_expected_metric_names_present(client):
    text = _get_prom(client)
    for name in METRIC_NAMES:
        assert name in text, name
    # up gauge must be present with value 1 even before any traffic.
    assert _lookup(text, "yt_rag_up", {}) == 1


def test_label_values_are_bounded(client):
    text = _get_prom(client)
    for labels, _ in _series(text, "yt_rag_requests_total"):
        assert set(labels) == {"endpoint", "method", "status"}
        assert labels["endpoint"] in PROM_ENDPOINTS
        assert labels["method"] in ALLOWED_METHODS
        assert labels["status"] in ALLOWED_STATUSES
    for labels, _ in _series(text, "yt_rag_errors_total"):
        assert set(labels) == {"endpoint", "method", "error_class"}
        assert labels["endpoint"] in PROM_ENDPOINTS
        assert labels["method"] in ALLOWED_METHODS
        assert labels["error_class"] in ALLOWED_ERROR_CLASSES
    for labels, _ in _series(text, "yt_rag_request_latency_seconds_bucket"):
        assert set(labels) == {"endpoint", "method", "le"}
        assert labels["endpoint"] in PROM_ENDPOINTS
        assert labels["method"] in ALLOWED_METHODS


def test_request_increments_requests_total(client):
    before = _get_prom(client)
    r = client.post(
        "/chat",
        json={"file": client.teal_fixture, "question": "how to optimize a linkedin profile"},
    )
    assert r.status_code == 200
    after = _get_prom(client)

    labels = {"endpoint": "/chat", "method": "POST", "status": "200"}
    assert _lookup(before, "yt_rag_requests_total", labels) in (None, 0.0)
    after_count = _lookup(after, "yt_rag_requests_total", labels)
    assert after_count is not None and after_count >= 1


def test_latency_metrics_exposed_after_chat(client):
    text = _get_prom(client)
    chat_labels = {"endpoint": "/chat", "method": "POST"}
    chat_buckets = [
        (labels, value)
        for labels, value in _series(text, "yt_rag_request_latency_seconds_bucket")
        if labels.get("endpoint") == "/chat"
    ]
    assert chat_buckets, text
    # Cumulative buckets end at +Inf and must match _count.
    inf = [value for labels, value in chat_buckets if labels["le"] == "+Inf"]
    assert inf
    count = _lookup(text, "yt_rag_request_latency_seconds_count", chat_labels)
    assert count is not None
    assert inf[-1] == count
    total = _lookup(text, "yt_rag_request_latency_seconds_sum", chat_labels)
    assert total is not None and total >= 0.0


def test_error_counter_increments_after_triggering_request(client):
    before = _get_prom(client)

    def error_total(text: str) -> float:
        return sum(value for _, value in _series(text, "yt_rag_errors_total"))

    assert client.post("/chat", json={"question": "hi"}).status_code == 422
    after = _get_prom(client)
    assert error_total(after) == error_total(before) + 1
    ingest = [
        labels
        for labels, _ in _series(after, "yt_rag_errors_total")
        if labels.get("error_class") == "IngestError"
    ]
    assert ingest, after
    assert ingest[-1]["endpoint"] == "/chat"
    assert ingest[-1]["method"] == "POST"


def test_json_metrics_endpoint_unchanged_contract(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    for key in ("in_process", "request_count", "error_count", "error_rate", "latency_ms"):
        assert key in body


def test_writer_is_deterministic_and_unit_testable():
    prom = PrometheusMetrics()
    prom.observe_request(endpoint="/chat", method="POST", status=200, duration_s=0.02)
    prom.observe_request(endpoint="/chat", method="POST", status=200, duration_s=0.4)
    prom.observe_error(endpoint="/chat", method="POST", error_class="IngestError")
    first = prom.render()
    second = prom.render()
    assert first == second
    assert 'yt_rag_requests_total{endpoint="/chat",method="POST",status="200"} 2' in first
    assert (
        'yt_rag_errors_total{endpoint="/chat",method="POST",error_class="IngestError"} 1' in first
    )
    assert (
        'yt_rag_request_latency_seconds_bucket{endpoint="/chat",method="POST",le="0.05"} 1' in first
    )
    assert (
        'yt_rag_request_latency_seconds_bucket{endpoint="/chat",method="POST",le="0.5"} 2' in first
    )
    assert (
        'yt_rag_request_latency_seconds_bucket{endpoint="/chat",method="POST",le="+Inf"} 2' in first
    )
    assert "yt_rag_up 1" in first


def test_retrieval_and_generation_latency_histograms_exposed(fresh_client):
    r = fresh_client.post(
        "/chat",
        json={"file": fresh_client.teal_fixture, "question": "how to optimize a linkedin profile"},
    )
    assert r.status_code == 200
    text = _get_prom(fresh_client)
    for name in ("yt_rag_retrieval_latency_seconds", "yt_rag_generation_latency_seconds"):
        labels_list = [labels for labels, _ in _series(text, f"{name}_count")]
        assert labels_list == [{"endpoint": "/chat"}]
        buckets = [(labels["le"], value) for labels, value in _series(text, f"{name}_bucket")]
        assert buckets[-1] == ("+Inf", 1.0)
        assert _lookup(text, f"{name}_sum", {"endpoint": "/chat"}) >= 0.0


def test_empty_results_total_increments_on_empty_result(teal_fixture):
    class EmptyRetriever:
        def retrieve(self, query: str):
            return []

    class NullProvider:
        def generate(self, question: str, retrieved) -> str:
            return "no retrieved chunks"

    class EmptyResultPipeline(RAGPipeline):
        def ingest_transcript(self, transcript):
            chunks = super().ingest_transcript(transcript)
            self.retriever = EmptyRetriever()
            return chunks

    def factory() -> RAGPipeline:
        return EmptyResultPipeline(embedder=HashEmbedder(dim=384), provider=NullProvider())

    app = create_app(pipeline_factory=factory)
    with TestClient(app) as c:
        assert _lookup(_get_prom(c), "yt_rag_empty_results_total", {"endpoint": "/chat"}) is None
        r = c.post("/chat", json={"file": str(teal_fixture), "question": "anything"})
        assert r.status_code == 200 and r.json()["retrieved"] == []
        assert _lookup(_get_prom(c), "yt_rag_empty_results_total", {"endpoint": "/chat"}) == 1.0


def test_mean_gauges_present_and_finite_after_chat(fresh_client):
    r = fresh_client.post(
        "/chat",
        json={"file": fresh_client.teal_fixture, "question": "how to optimize a linkedin profile"},
    )
    assert r.status_code == 200
    text = _get_prom(fresh_client)
    top = _lookup(text, "yt_rag_mean_top_score", {"endpoint": "/chat"})
    assert top is not None
    assert 0.0 <= top <= 1.0
    mrc = _lookup(text, "yt_rag_mean_retrieved_count", {"endpoint": "/chat"})
    assert mrc is not None and mrc >= 1.0
    # HELP text must flag both as proxies, not answer quality.
    assert "PROXY metric" in text
    assert "NOT answer quality" in text


def test_question_length_histogram_exposed(fresh_client):
    r = fresh_client.post(
        "/chat",
        json={"file": fresh_client.teal_fixture, "question": "how to optimize a linkedin profile"},
    )
    assert r.status_code == 200
    text = _get_prom(fresh_client)
    labels_list = [labels for labels, _ in _series(text, "yt_rag_question_length_chars_count")]
    assert labels_list == [{"endpoint": "/chat"}]
    buckets = [
        (labels["le"], value)
        for labels, value in _series(text, "yt_rag_question_length_chars_bucket")
    ]
    assert buckets[0] == ("10", 0.0)
    assert buckets[-1] == ("+Inf", 1.0)
    assert _lookup(text, "yt_rag_question_length_chars_sum", {"endpoint": "/chat"}) > 0.0


def test_provider_mode_has_bounded_label_values(client):
    text = _get_prom(client)
    modes = [labels["mode"] for labels, _ in _series(text, "yt_rag_provider_mode")]
    assert modes == ["stub"]  # stub app: exactly one bounded mode series
    assert all(m in {"stub", "openai_compatible", "unknown"} for m in modes)


def test_provider_mode_unknown_fallback_is_bounded(teal_fixture):
    class NullProvider:
        def generate(self, question: str, retrieved) -> str:
            return "answer"

    app = create_app(embedder=HashEmbedder(dim=384), provider=NullProvider())
    with TestClient(app) as c:
        r = c.post("/chat", json={"file": str(teal_fixture), "question": "anything"})
        assert r.status_code == 200
        modes = [labels["mode"] for labels, _ in _series(_get_prom(c), "yt_rag_provider_mode")]
        assert modes == ["unknown"]


def test_no_unbounded_label_growth_after_varied_requests(client):
    questions = ["a", "x" * 300, "another question about nothing in particular", "12345"]
    for q in questions:
        r = client.post("/chat", json={"file": client.teal_fixture, "question": q})
        assert r.status_code == 200
    assert client.post("/chat", json={"question": "hi"}).status_code == 422
    text = _get_prom(client)
    endpoints = set()
    for family in (
        "yt_rag_requests_total",
        "yt_rag_errors_total",
        "yt_rag_request_latency_seconds_bucket",
        "yt_rag_retrieval_latency_seconds_bucket",
        "yt_rag_generation_latency_seconds_bucket",
        "yt_rag_question_length_chars_bucket",
        "yt_rag_empty_results_total",
        "yt_rag_mean_top_score",
        "yt_rag_mean_retrieved_count",
    ):
        for labels, _ in _series(text, family):
            endpoints.add(labels.get("endpoint"))
    assert endpoints <= PROM_ENDPOINTS
    # The only expected scrape-to-scrape change is this scrape recording
    # itself afterwards (one /metrics/prometheus sample); determinism of the
    # writer for identical state is covered by the unit test below.


def test_writer_rejects_unbounded_label_values():
    prom = PrometheusMetrics()
    with pytest.raises(ValueError):
        prom.observe_request(
            endpoint="/chat?question=leak", method="POST", status=200, duration_s=0.01
        )
    with pytest.raises(ValueError):
        prom.observe_error(endpoint="/chat", method="POST", error_class="SomeRandomError: boom")
