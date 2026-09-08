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

# Phase 3: bounded provider-mode vocabulary for yt_rag_provider_mode. Derived
# from the generation provider class name; anything unrecognized maps to
# "unknown" so the label can never grow unbounded.
PROM_PROVIDER_MODES: frozenset[str] = frozenset({"stub", "openai_compatible", "unknown"})

# Phase 3: question-length buckets (characters). Chat questions are one short
# sentence; buckets span that to the few-thousand-character tail, then +Inf.
QUESTION_LENGTH_BUCKETS: tuple[float, ...] = (10, 50, 100, 200, 500, 1000, 2000)


def provider_mode_name(provider: object) -> str:
    """Map a generation provider instance to a bounded mode label value."""
    name = type(provider).__name__
    return {
        "StubProvider": "stub",
        "OpenAICompatibleProvider": "openai_compatible",
    }.get(name, "unknown")


class _Histogram:
    """Cumulative bucket counts + sum/count for one labeled histogram series."""

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = tuple(sorted(buckets))
        self.counts = [0] * len(self.buckets)  # cumulative per bucket
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        value = max(0.0, float(value))
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1
        self.sum += value
        self.count += 1


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
        # Phase 3 domain families (all /chat only, endpoint label only).
        self.retrieval_latency: dict[str, _Histogram] = {}
        self.generation_latency: dict[str, _Histogram] = {}
        self.question_length: dict[str, _Histogram] = {}
        self.empty_results: dict[str, int] = {}
        # Rolling means mirroring the JSON snapshot semantics:
        # mean_top_score averages top scores over non-empty successes;
        # mean_retrieved_count averages retrieved counts over all successes.
        self.top_score_sum: dict[str, float] = {}
        self.top_score_n: dict[str, int] = {}
        self.retrieved_count_sum: dict[str, float] = {}
        self.retrieved_count_n: dict[str, int] = {}
        self.provider_mode: str | None = None

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

    def _domain_histogram(
        self, store: dict[str, _Histogram], buckets: tuple[float, ...], endpoint: str
    ) -> _Histogram:
        self._check(endpoint, PROM_ENDPOINTS, "endpoint")
        hist = store.get(endpoint)
        if hist is None:
            hist = _Histogram(buckets)
            store[endpoint] = hist
        return hist

    def observe_retrieval_latency(self, *, endpoint: str, duration_s: float) -> None:
        """Record retrieval latency for yt_rag_retrieval_latency_seconds."""
        self._domain_histogram(self.retrieval_latency, LATENCY_BUCKETS, endpoint).observe(
            duration_s
        )

    def observe_generation_latency(self, *, endpoint: str, duration_s: float) -> None:
        """Record generation latency for yt_rag_generation_latency_seconds."""
        self._domain_histogram(self.generation_latency, LATENCY_BUCKETS, endpoint).observe(
            duration_s
        )

    def observe_question_length(self, *, endpoint: str, chars: int) -> None:
        """Record question length for yt_rag_question_length_chars."""
        self._domain_histogram(self.question_length, QUESTION_LENGTH_BUCKETS, endpoint).observe(
            float(chars)
        )

    def observe_empty_result(self, *, endpoint: str) -> None:
        """Record one successful chat with zero retrieved chunks."""
        self._check(endpoint, PROM_ENDPOINTS, "endpoint")
        self.empty_results[endpoint] = self.empty_results.get(endpoint, 0) + 1

    def observe_success_quality(
        self, *, endpoint: str, retrieved_count: int, top_score: float | None
    ) -> None:
        """Accumulate rolling-mean proxy inputs (mirrors JSON snapshot rules).

        mean_top_score averages top scores over NON-EMPTY successes only;
        mean_retrieved_count averages retrieved counts over ALL successes.
        """
        self._check(endpoint, PROM_ENDPOINTS, "endpoint")
        self.retrieved_count_sum[endpoint] = self.retrieved_count_sum.get(endpoint, 0.0) + float(
            retrieved_count
        )
        self.retrieved_count_n[endpoint] = self.retrieved_count_n.get(endpoint, 0) + 1
        if top_score is not None:
            self.top_score_sum[endpoint] = self.top_score_sum.get(endpoint, 0.0) + float(top_score)
            self.top_score_n[endpoint] = self.top_score_n.get(endpoint, 0) + 1

    def set_provider_mode(self, mode: str) -> None:
        """Record the active generation provider mode (bounded vocabulary)."""
        if mode not in PROM_PROVIDER_MODES:
            raise ValueError(f"provider mode {mode!r} is outside the bounded vocabulary")
        self.provider_mode = mode

    def _render_histogram_family(
        self, lines: list[str], name: str, help_text: str, store: dict[str, _Histogram]
    ) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} histogram")
        for endpoint in sorted(store):
            hist = store[endpoint]
            base = f'{{endpoint="{_escape_label_value(endpoint)}"}}'
            for bound, count in zip(hist.buckets, hist.counts):  # counts stored cumulative
                lines.append(
                    f'{name}_bucket{base[:-1]},le="{_format_label_float(bound)}"}} {count}'
                )
            lines.append(f'{name}_bucket{base[:-1]},le="+Inf"}} {hist.count}')
            lines.append(f"{name}_sum{base} {hist.sum:.6f}")
            lines.append(f"{name}_count{base} {hist.count}")

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

        # Phase 3: app-specific families (all /chat only, low-cardinality).
        empty = f"{PROM_PREFIX}_empty_results_total"
        lines.append(
            f"# HELP {empty} Successful requests with zero retrieved chunks; divide by requests_total to approximate the empty-result rate."
        )
        lines.append(f"# TYPE {empty} counter")
        for endpoint in sorted(self.empty_results):
            lines.append(
                f'{empty}{{endpoint="{_escape_label_value(endpoint)}"}} {self.empty_results[endpoint]}'
            )

        self._render_histogram_family(
            lines,
            f"{PROM_PREFIX}_retrieval_latency_seconds",
            "Retrieval latency in seconds (time to run top-k search).",
            self.retrieval_latency,
        )
        self._render_histogram_family(
            lines,
            f"{PROM_PREFIX}_generation_latency_seconds",
            "Generation latency in seconds (time spent in the LLM provider).",
            self.generation_latency,
        )
        self._render_histogram_family(
            lines,
            f"{PROM_PREFIX}_question_length_chars",
            "Question length in characters.",
            self.question_length,
        )

        top = f"{PROM_PREFIX}_mean_top_score"
        lines.append(
            f"# HELP {top} PROXY metric: rolling mean of the top retrieval score over non-empty successes. NOT answer quality."
        )
        lines.append(f"# TYPE {top} gauge")
        for endpoint in sorted(self.top_score_n):
            if self.top_score_n[endpoint]:
                lines.append(
                    f'{top}{{endpoint="{_escape_label_value(endpoint)}"}} {self.top_score_sum[endpoint] / self.top_score_n[endpoint]:.4f}'
                )

        mrc = f"{PROM_PREFIX}_mean_retrieved_count"
        lines.append(
            f"# HELP {mrc} PROXY metric: rolling mean number of retrieved chunks over successful requests. NOT answer quality."
        )
        lines.append(f"# TYPE {mrc} gauge")
        for endpoint in sorted(self.retrieved_count_n):
            if self.retrieved_count_n[endpoint]:
                lines.append(
                    f'{mrc}{{endpoint="{_escape_label_value(endpoint)}"}} {self.retrieved_count_sum[endpoint] / self.retrieved_count_n[endpoint]:.4f}'
                )

        pm = f"{PROM_PREFIX}_provider_mode"
        lines.append(
            f"# HELP {pm} Active generation provider mode (stub / openai_compatible / unknown); 1 for the active mode."
        )
        lines.append(f"# TYPE {pm} gauge")
        if self.provider_mode is not None:
            lines.append(f'{pm}{{mode="{self.provider_mode}"}} 1')

        lines.append(f"# HELP {PROM_PREFIX}_up Whether the app is serving (1 = up).")
        lines.append(f"# TYPE {PROM_PREFIX}_up gauge")
        lines.append(f"{PROM_PREFIX}_up 1")
        return "\n".join(lines) + "\n"


def _format_label_float(value: float) -> str:
    """Render a bucket bound as a short, stable float string."""
    text = f"{value:g}"
    return text
