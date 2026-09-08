"""Stage 5 serving observability: structured JSON request logs + in-process metrics.

Scope (stated honestly):

- Metrics are plain in-process counters. They reset when the process restarts.
  There is no metrics store, no Prometheus/Grafana stack, and no alerting.
- The retrieval-quality signal is a PROXY (empty-result rate and top retrieval
  score), not a quality measure. Stage 2 groundedness is a stub and the
  qualitative LLM review is still open; nothing here relabels that.
- Logs never contain API keys, prompts, answers, or retrieved text — only
  counts, scores, ids, and timings.
- GET /metrics/prometheus serves the same in-process counters in Prometheus
  text exposition format (stdlib-only writer, no prometheus_client). Generic
  families only (requests/errors/latency/up); no domain metrics yet.
"""

from __future__ import annotations

import json
import logging
import math
import sys
import uuid
from collections import deque
from datetime import UTC, datetime

LOGGER_NAME = "yt_rag.observability"

# Prometheus metric prefix for this app.
PROM_PREFIX = "yt_rag"

# Documented latency buckets (seconds) for yt_rag_request_latency_seconds.
# Chat calls an LLM (sub-second with the offline stub, seconds with a real
# provider); ingestion/embedding add model work. Buckets span the fast stub
# path (~ms) to a slow real-provider request (~10s), then +Inf.
LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)

# Bounded label vocabularies for the Prometheus families. endpoint is a route
# template, never a full URL; status covers only statuses this app actually
# returns; error_class is the app's exception-class vocabulary (see errors.py
# and the handlers in app.py). Nothing user-supplied is ever a label value.
PROM_ENDPOINTS: frozenset[str] = frozenset({"/health", "/metrics", "/metrics/prometheus", "/chat"})
PROM_STATUSES: frozenset[int] = frozenset({200, 400, 404, 422, 500})
PROM_ERROR_CLASSES: frozenset[str] = frozenset(
    {
        "IngestError",
        "GenerationError",
        "IndexStateError",
        "RuntimeError",
        "ValueError",
        "Unhandled",
    }
)

# Bounded sample buffers: enough history for percentile summaries without
# growing without bound in a long-lived process.
_MAX_SAMPLES = 2048


def get_logger() -> logging.Logger:
    """Return the observability logger, writing one JSON object per line to stdout."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def new_request_id() -> str:
    """Short opaque request id. Random, carries no user data."""
    return uuid.uuid4().hex[:12]


def log_request(logger: logging.Logger, **fields) -> None:
    """Emit one structured JSON log line. Values must already be log-safe."""
    payload = {
        "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
        "level": "INFO",
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=False))


def _summary(samples: deque[float]) -> dict:
    """Latency summary in ms: count, mean, p50, p95, max."""
    if not samples:
        return {"count": 0}
    ordered = sorted(samples)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, math.ceil(p * n) - 1))
        return round(ordered[idx], 2)

    return {
        "count": n,
        "mean_ms": round(sum(ordered) / n, 2),
        "p50_ms": pct(0.50),
        "p95_ms": pct(0.95),
        "max_ms": round(ordered[-1], 2),
    }


class Metrics:
    """In-process request counters backing GET /metrics.

    Deliberately minimal: counters and bounded latency samples, reset on
    restart. request_count covers /chat traffic only; probe endpoints
    (/health, /metrics) are logged but not counted.
    """

    def __init__(self) -> None:
        self.started_at = datetime.now(UTC).isoformat(timespec="seconds")
        self.request_count = 0
        self.error_count = 0
        self.error_classes: dict[str, int] = {}
        self.total_ms: deque[float] = deque(maxlen=_MAX_SAMPLES)
        self.retrieval_ms: deque[float] = deque(maxlen=_MAX_SAMPLES)
        self.generation_ms: deque[float] = deque(maxlen=_MAX_SAMPLES)
        self.empty_results = 0
        self.retrieved_counts: deque[int] = deque(maxlen=_MAX_SAMPLES)
        self.top_scores: deque[float] = deque(maxlen=_MAX_SAMPLES)

    def record_success(
        self,
        *,
        total_ms: float,
        retrieval_ms: float,
        generation_ms: float,
        retrieved_count: int,
        top_score: float | None,
    ) -> None:
        self.request_count += 1
        self.total_ms.append(total_ms)
        self.retrieval_ms.append(retrieval_ms)
        self.generation_ms.append(generation_ms)
        self.retrieved_counts.append(retrieved_count)
        if retrieved_count == 0:
            self.empty_results += 1
        if top_score is not None:
            self.top_scores.append(top_score)

    def record_error(self, *, total_ms: float, error_class: str) -> None:
        self.request_count += 1
        self.error_count += 1
        self.error_classes[error_class] = self.error_classes.get(error_class, 0) + 1
        self.total_ms.append(total_ms)

    def snapshot(self) -> dict:
        successes = len(self.retrieved_counts)
        proxy = {
            "label": (
                "PROXY - empty-result rate and top retrieval score; "
                "NOT a quality measure (Stage 2 groundedness is a stub, "
                "qualitative LLM review is open)"
            ),
            "empty_result_count": self.empty_results,
            "empty_result_rate": round(self.empty_results / successes, 4) if successes else 0.0,
            "mean_top_score": (
                round(sum(self.top_scores) / len(self.top_scores), 4) if self.top_scores else None
            ),
            "mean_retrieved_count": (
                round(sum(self.retrieved_counts) / successes, 2) if successes else 0.0
            ),
        }
        return {
            "in_process": True,
            "note": (
                "Counters live in this process only and reset on restart. "
                "retrieval_quality_proxy is a PROXY, not a quality measure."
            ),
            "started_at": self.started_at,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate": round(self.error_count / self.request_count, 4)
            if self.request_count
            else 0.0,
            "error_classes": dict(sorted(self.error_classes.items())),
            "latency_ms": {
                "total": _summary(self.total_ms),
                "retrieval": _summary(self.retrieval_ms),
                "generation": _summary(self.generation_ms),
            },
            "retrieval_quality_proxy": proxy,
        }


def _escape_label_value(value: str) -> str:
    """Escape a label value per the Prometheus text exposition rules."""
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_labels(labels: dict[str, str]) -> str:
    inner = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in labels.items())
    return f"{{{inner}}}" if inner else ""


class PrometheusMetrics:
    """In-process state for the generic Prometheus families (stdlib only).

    Families (prefix yt_rag_):
      - yt_rag_requests_total counter {endpoint, method, status}
      - yt_rag_errors_total counter {endpoint, method, error_class}
      - yt_rag_request_latency_seconds histogram {endpoint, method}
      - yt_rag_up gauge

    Label values are validated against the bounded vocabularies above; an out-
    of-vocabulary value raises ValueError instead of silently growing
    cardinality. Rendering is deterministic: families and series are sorted.
    """

    def __init__(self, buckets: tuple[float, ...] = LATENCY_BUCKETS) -> None:
        self.buckets = tuple(sorted(buckets))
        self.requests: dict[tuple[str, str, int], int] = {}
        self.errors: dict[tuple[str, str, str], int] = {}
        # (endpoint, method) -> per-bucket cumulative counts, plus sum/count.
        self.latency_buckets: dict[tuple[str, str], list[int]] = {}
        self.latency_sum: dict[tuple[str, str], float] = {}
        self.latency_count: dict[tuple[str, str], int] = {}

    @staticmethod
    def _check(value: str, allowed: frozenset, kind: str) -> str:
        if value not in allowed:
            raise ValueError(f"{kind} {value!r} is outside the bounded vocabulary")
        return value

    def observe_request(
        self, *, endpoint: str, method: str, status: int, duration_s: float
    ) -> None:
        """Record one served request: requests_total, errors_total, latency."""
        self._check(endpoint, PROM_ENDPOINTS, "endpoint")
        self._check(status, PROM_STATUSES, "status")
        key = (endpoint, method, status)
        self.requests[key] = self.requests.get(key, 0) + 1
        bkey = (endpoint, method)
        if bkey not in self.latency_buckets:
            self.latency_buckets[bkey] = [0] * len(self.buckets)
            self.latency_sum[bkey] = 0.0
            self.latency_count[bkey] = 0
        counts = self.latency_buckets[bkey]
        duration_s = max(0.0, float(duration_s))
        for i, bound in enumerate(self.buckets):
            if duration_s <= bound:
                counts[i] += 1
        self.latency_sum[bkey] += duration_s
        self.latency_count[bkey] += 1

    def observe_error(self, *, endpoint: str, method: str, error_class: str) -> None:
        """Record one classified error for yt_rag_errors_total."""
        self._check(endpoint, PROM_ENDPOINTS, "endpoint")
        self._check(error_class, PROM_ERROR_CLASSES, "error_class")
        key = (endpoint, method, error_class)
        self.errors[key] = self.errors.get(key, 0) + 1

    def render(self) -> str:
        """Deterministic Prometheus text exposition (version 0.0.4)."""
        lines: list[str] = []
        for name, help_text in (
            (f"{PROM_PREFIX}_requests_total", "Total HTTP requests served."),
            (f"{PROM_PREFIX}_errors_total", "Total classified request errors."),
        ):
            series = self.requests if name.endswith("requests_total") else self.errors
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} counter")
            for key in sorted(series, key=repr):
                value = series[key]
                if name.endswith("requests_total"):
                    endpoint, method, status = key
                    labels = _format_labels(
                        {"endpoint": endpoint, "method": method, "status": str(status)}
                    )
                else:
                    endpoint, method, error_class = key
                    labels = _format_labels(
                        {"endpoint": endpoint, "method": method, "error_class": error_class}
                    )
                lines.append(f"{name}{labels} {value}")

        hist = f"{PROM_PREFIX}_request_latency_seconds"
        lines.append(f"# HELP {hist} Request latency in seconds.")
        lines.append(f"# TYPE {hist} histogram")
        for bkey in sorted(self.latency_count, key=repr):
            endpoint, method = bkey
            base = _format_labels({"endpoint": endpoint, "method": method})
            counts = self.latency_buckets[bkey]
            # observe_request already stores cumulative per-bucket counts.
            for bound, cumulative in zip(self.buckets, counts):
                lines.append(
                    f'{hist}_bucket{base[:-1]},le="{_format_label_float(bound)}"}} {cumulative}'
                )
            lines.append(f'{hist}_bucket{base[:-1]},le="+Inf"}} {self.latency_count[bkey]}')
            lines.append(f"{hist}_sum{base} {self.latency_sum[bkey]:.6f}")
            lines.append(f"{hist}_count{base} {self.latency_count[bkey]}")

        lines.append(f"# HELP {PROM_PREFIX}_up Whether the app is serving (1 = up).")
        lines.append(f"# TYPE {PROM_PREFIX}_up gauge")
        lines.append(f"{PROM_PREFIX}_up 1")
        return "\n".join(lines) + "\n"


def _format_label_float(value: float) -> str:
    """Render a bucket bound as a short, stable float string."""
    text = f"{value:g}"
    return text
