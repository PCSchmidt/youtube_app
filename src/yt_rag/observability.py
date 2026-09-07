"""Stage 5 serving observability: structured JSON request logs + in-process metrics.

Scope (stated honestly):

- Metrics are plain in-process counters. They reset when the process restarts.
  There is no metrics store, no Prometheus/Grafana stack, and no alerting.
- The retrieval-quality signal is a PROXY (empty-result rate and top retrieval
  score), not a quality measure. Stage 2 groundedness is a stub and the
  qualitative LLM review is still open; nothing here relabels that.
- Logs never contain API keys, prompts, answers, or retrieved text — only
  counts, scores, ids, and timings.
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
