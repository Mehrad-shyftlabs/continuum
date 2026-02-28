#!/usr/bin/env python3
"""
CLI for SDK Feature Test playground.

Run full pipeline:  python -m playground.sdk_feature_test run --scenario full
Health only:        python -m playground.sdk_feature_test run --scenario health
Workflow only:     python -m playground.sdk_feature_test run --scenario workflow-only
No Temporal:       python -m playground.sdk_feature_test run --scenario no-temporal

Interactive:       python -m playground.sdk_feature_test
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure project root and src on path
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))

from orchestrator import get_logger, setup_logging
from orchestrator.agent.types import generate_run_id

from playground.sdk_feature_test.config import get_config
from playground.sdk_feature_test.pipeline import (
    run_full_pipeline,
    run_health_only,
    run_hitl_pipeline,
    run_no_temporal,
    run_workflow_only,
)

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SDK Feature Test: multi-agent pipeline that exercises all Orchestrator SDK features.",
    )
    sub = parser.add_subparsers(dest="command", help="Command")
    run_parser = sub.add_parser("run", help="Run a scenario")
    run_parser.add_argument(
        "--scenario",
        choices=["full", "workflow-only", "no-temporal", "health", "hitl"],
        default="full",
        help="full = all features; workflow-only = core + router/seq/par/loop; no-temporal = full without Temporal; health = health checks only; hitl = Temporal with human-in-the-loop approval",
    )
    run_parser.add_argument(
        "--user-id",
        default=None,
        help="User ID for session/memory (default: generated)",
    )
    run_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging",
    )
    return parser.parse_args()


async def _run_scenario(scenario: str, user_id: str | None, verbose: bool) -> int:
    cfg = get_config()  # respects TEMPORAL_ENABLED etc. from env
    if verbose:
        setup_logging(level="DEBUG")
    uid = user_id or f"sdk-test-{generate_run_id()[-8:]}"
    session_id: str | None = None

    if scenario == "health":
        ok = await run_health_only(cfg)
        return 0 if ok else 1
    if scenario == "workflow-only":
        ok = await run_workflow_only(cfg, uid, session_id)
        return 0 if ok else 1
    if scenario == "no-temporal":
        ok = await run_no_temporal(cfg, uid, session_id)
        return 0 if ok else 1
    if scenario == "hitl":
        ok = await run_hitl_pipeline(cfg, uid, session_id)
        return 0 if ok else 1
    if scenario == "full":
        ok = await run_full_pipeline(cfg, uid, session_id, include_temporal=cfg.enable_temporal)
        return 0 if ok else 1
    return 1


def _interactive_menu() -> int:
    print("\nSDK Feature Test - Research & Report Pipeline")
    print(" 1. Health only")
    print(" 2. Workflow only (router, sequential, parallel, loop)")
    print(" 3. Full (no Temporal)")
    print(" 4. Full (with Temporal if enabled)")
    print(" 5. Human-in-the-Loop (Temporal + approval gate)")
    print(" q. Quit")
    choice = input("Choice [1-5/q]: ").strip().lower()
    if choice == "q":
        return 0
    scenario_map = {
        "1": "health",
        "2": "workflow-only",
        "3": "no-temporal",
        "4": "full",
        "5": "hitl",
    }
    scenario = scenario_map.get(choice, "health")
    return asyncio.run(_run_scenario(scenario, None, verbose=True))


def main() -> int:
    args = _parse_args()
    if args.command != "run":
        return _interactive_menu()
    return asyncio.run(_run_scenario(args.scenario, args.user_id, getattr(args, "verbose", False)))


if __name__ == "__main__":
    sys.exit(main())
