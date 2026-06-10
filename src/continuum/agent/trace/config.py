"""
Trace feature configuration helpers.

Centralizes reading the ``DECISION_TRACE_*`` settings and building the configured
:class:`TraceStore`. The store is cached so each run reuses one Redis connection.
"""

from __future__ import annotations

from functools import lru_cache

from continuum.agent.trace.store import (
    InMemoryTraceStore,
    NullTraceStore,
    RedisTraceStore,
    TraceStore,
)
from continuum.agent.trace.types import TraceDetail


def is_trace_enabled() -> bool:
    from continuum.config import settings

    return bool(settings.decision_trace_enabled)


def trace_detail() -> TraceDetail:
    from continuum.config import settings

    return TraceDetail(settings.decision_trace_detail)


def checkpoint_enabled() -> bool:
    from continuum.config import settings

    return bool(settings.decision_trace_checkpoint)


@lru_cache(maxsize=1)
def get_trace_store() -> TraceStore:
    """Build the configured trace store once (cached).

    Falls back to a :class:`NullTraceStore` if a Redis backend is requested but
    its client cannot be constructed — persistence must never break a run.
    """
    from continuum.config import settings

    backend = settings.decision_trace_store
    if backend == "memory":
        return InMemoryTraceStore()
    if backend == "null":
        return NullTraceStore()
    try:
        return RedisTraceStore(ttl_seconds=settings.decision_trace_ttl_days * 24 * 3600)
    except Exception:
        return NullTraceStore()
