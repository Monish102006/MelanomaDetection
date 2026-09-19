"""
Observability & Performance Metrics — Melanoma Agentic RAG v4.0

Provides:
- Structured JSON logging
- Latency tracking (retrieval, LLM, end-to-end)
- Token usage tracking (prompt/completion/total)
- Performance metrics collection and aggregation
- Request-scoped tracing with unique request IDs

Usage:
    from observability import MetricsCollector, create_logger, timed

    logger = create_logger(__name__)
    metrics = MetricsCollector()

    with metrics.track("dense_search"):
        results = vector_store.search(...)

    metrics.record_tokens(prompt=100, completion=50)
    summary = metrics.summary()
"""

import time
import uuid
import json
import logging
from typing import Any, Dict, Optional, List
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone


# ============================================================
# Structured JSON Logger
# ============================================================

class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach extra structured fields if present
        if hasattr(record, "structured_data"):
            log_entry["data"] = record.structured_data

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])

        return json.dumps(log_entry, default=str)


def create_logger(
    name: str,
    level: int = logging.INFO,
    use_json: bool = False
) -> logging.Logger:
    """
    Create a logger with optional JSON formatting.

    Parameters
    ----------
    name : str
        Logger name (typically __name__)
    level : int
        Logging level
    use_json : bool
        If True, output structured JSON logs.
        If False, output human-readable text logs.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()

        if use_json:
            handler.setFormatter(JSONFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            ))

        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


# ============================================================
# Timing Utilities
# ============================================================

@dataclass
class TimingRecord:
    """A single timing measurement."""
    operation: str
    duration_ms: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# Token Usage Tracking
# ============================================================

@dataclass
class TokenUsage:
    """Tracks token consumption for an LLM call."""
    operation: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    timestamp: str = ""


# ============================================================
# Metrics Collector (Per-Request)
# ============================================================

class MetricsCollector:
    """
    Collects performance metrics for a single pipeline execution.
    Create a new instance per request/invocation for isolation.

    Usage:
        metrics = MetricsCollector(request_id="REQ-001")

        with metrics.track("dense_search"):
            results = search(...)

        metrics.record_tokens("evaluation", prompt=100, completion=50)

        print(metrics.summary())
    """

    def __init__(self, request_id: Optional[str] = None):
        self.request_id = request_id or f"req-{uuid.uuid4().hex[:8]}"
        self.start_time = time.perf_counter()
        self.timings: List[TimingRecord] = []
        self.token_usage: List[TokenUsage] = []
        self.counters: Dict[str, int] = {}
        self._logger = logging.getLogger("observability.metrics")

    @contextmanager
    def track(self, operation: str, **metadata):
        """
        Context manager to time an operation.

        Usage:
            with metrics.track("reranking", num_docs=10):
                reranked = rerank(query, docs)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            record = TimingRecord(
                operation=operation,
                duration_ms=round(duration_ms, 2),
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata=metadata
            )
            self.timings.append(record)
            self._logger.debug(
                f"[{self.request_id}] {operation}: {duration_ms:.2f}ms"
            )

    def record_tokens(
        self,
        operation: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = ""
    ):
        """Record token usage for an LLM call."""
        total = prompt_tokens + completion_tokens
        usage = TokenUsage(
            operation=operation,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            model=model,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        self.token_usage.append(usage)

    def increment(self, counter_name: str, amount: int = 1):
        """Increment a named counter."""
        self.counters[counter_name] = self.counters.get(counter_name, 0) + amount

    def summary(self) -> Dict[str, Any]:
        """
        Generate a complete metrics summary for the request.

        Returns a dictionary with:
        - request_id
        - total_duration_ms
        - timing breakdown by operation
        - token usage summary
        - counters
        """
        total_duration_ms = (time.perf_counter() - self.start_time) * 1000

        # Aggregate timings by operation
        timing_breakdown = {}
        for t in self.timings:
            if t.operation not in timing_breakdown:
                timing_breakdown[t.operation] = {
                    "total_ms": 0.0,
                    "count": 0,
                    "details": []
                }
            timing_breakdown[t.operation]["total_ms"] += t.duration_ms
            timing_breakdown[t.operation]["count"] += 1
            timing_breakdown[t.operation]["details"].append({
                "duration_ms": t.duration_ms,
                "metadata": t.metadata
            })

        # Round totals
        for op in timing_breakdown:
            timing_breakdown[op]["total_ms"] = round(
                timing_breakdown[op]["total_ms"], 2
            )

        # Token summary
        total_prompt = sum(u.prompt_tokens for u in self.token_usage)
        total_completion = sum(u.completion_tokens for u in self.token_usage)
        total_tokens = sum(u.total_tokens for u in self.token_usage)

        token_summary = {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_tokens,
            "calls": [
                {
                    "operation": u.operation,
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "total_tokens": u.total_tokens,
                    "model": u.model
                }
                for u in self.token_usage
            ]
        }

        return {
            "request_id": self.request_id,
            "total_duration_ms": round(total_duration_ms, 2),
            "timing_breakdown": timing_breakdown,
            "token_usage": token_summary,
            "counters": self.counters
        }


# ============================================================
# Global Metrics Aggregator (Cross-Request)
# ============================================================

class GlobalMetricsAggregator:
    """
    Aggregates metrics across multiple requests for the
    /metrics API endpoint. Thread-safe via simple append-only
    list with bounded history.
    """

    def __init__(self, max_history: int = 100):
        self.max_history = max_history
        self._history: List[Dict[str, Any]] = []
        self._total_requests: int = 0
        self._total_errors: int = 0

    def record(self, metrics_summary: Dict[str, Any]):
        """Record a completed request's metrics."""
        self._total_requests += 1
        self._history.append(metrics_summary)

        # Bounded history
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

    def record_error(self):
        """Record a failed request."""
        self._total_errors += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregated statistics across recent requests."""
        if not self._history:
            return {
                "total_requests": self._total_requests,
                "total_errors": self._total_errors,
                "recent_requests": 0,
                "avg_latency_ms": 0,
                "p50_latency_ms": 0,
                "p95_latency_ms": 0,
                "avg_tokens_per_request": 0
            }

        latencies = sorted(
            h["total_duration_ms"] for h in self._history
        )
        tokens = [
            h.get("token_usage", {}).get("total_tokens", 0)
            for h in self._history
        ]

        n = len(latencies)

        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "recent_requests": n,
            "avg_latency_ms": round(sum(latencies) / n, 2),
            "p50_latency_ms": round(latencies[n // 2], 2),
            "p95_latency_ms": round(latencies[int(n * 0.95)], 2),
            "min_latency_ms": round(latencies[0], 2),
            "max_latency_ms": round(latencies[-1], 2),
            "avg_tokens_per_request": round(sum(tokens) / n, 2) if tokens else 0,
            "total_tokens_consumed": sum(tokens)
        }


# ── Global aggregator singleton ───────────────────────────────

global_metrics = GlobalMetricsAggregator()
