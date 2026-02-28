"""
Configuration for SDK Feature Test playground.

Feature flags and limits for Core, Memory, Session, MCP, Temporal, etc.
Graceful degradation: if Redis/Qdrant/Langfuse are down, session/memory/tracing
are skipped and the pipeline continues.
"""

import os
from dataclasses import dataclass


def _env_bool(key: str, default: bool = False) -> bool:
    val = os.environ.get(key, "").strip().lower()
    return val in ("1", "true", "yes") if val else default


@dataclass
class SDKFeatureTestConfig:
    """Configuration for the Research & Report pipeline."""

    # Feature flags
    enable_memory: bool = True
    enable_session: bool = True
    enable_temporal: bool = False  # Set True or TEMPORAL_ENABLED=1 when Temporal is configured
    use_fake_mcp: bool = True  # True = in-process FakeMCPServer; False = external MCP URL
    mcp_server_url: str = "https://mcp.agentfly.shyftops.io/mcp"
    mcp_timeout: float = 30.0
    mcp_sse_timeout: float = 300.0

    # Models
    default_model: str = "gemini/gemini-2.5-flash"
    router_temperature: float = 0.3
    agent_temperature: float = 0.5
    max_turns: int = 15

    # Pipeline limits
    loop_max_iterations: int = 3
    parallel_timeout_per_agent: int = 120

    # Memory
    memory_search_limit: int = 5

    # Session
    session_ttl: int = 3600 * 24

    # Observability
    enable_tracing: bool = True

    # Container / lifecycle
    fail_on_unhealthy: bool = False
    verify_connections: bool = True


def get_config() -> SDKFeatureTestConfig:
    """Build config with optional env overrides (e.g. TEMPORAL_ENABLED=1)."""
    return SDKFeatureTestConfig(
        enable_temporal=_env_bool("TEMPORAL_ENABLED", False),
    )


default_config = SDKFeatureTestConfig()
