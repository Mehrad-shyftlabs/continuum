"""
Agents for the Research & Report pipeline.

- Router: research, summarize, qa
- Sequential: researcher (with MCP) -> writer (structured output) -> editor
- Parallel: analyst_a, analyst_b
- Loop: refiner
- Handoff: fact_checker
"""

from playground.sdk_feature_test.agents.build import (
    build_fact_checker_agent,
    build_parallel_agents,
    build_researcher_agent,
    build_router_agent,
    build_sequential_agent,
    build_loop_agent,
    build_writer_agent,
    build_editor_agent,
    build_analyst_agents,
    build_refiner_agent,
)

__all__ = [
    "build_router_agent",
    "build_researcher_agent",
    "build_writer_agent",
    "build_editor_agent",
    "build_sequential_agent",
    "build_analyst_agents",
    "build_parallel_agents",
    "build_refiner_agent",
    "build_loop_agent",
    "build_fact_checker_agent",
]
