"""Unit tests for the ``continuum`` CLI (infra orchestration).

These cover the pure, side-effect-free logic:
  * profile specifications (which compose profiles + which env each tier implies)
  * the ``docker compose`` argv builder
  * bundled compose-file resolution
  * the idempotent managed-``.env`` writer (write / re-run / preserve-user / warn-on-conflict)

The subprocess and Docker daemon are never touched here — that is covered by the
integration suite (``tests/integration/test_cli_compose.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from continuum import cli

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Profile specifications
# ---------------------------------------------------------------------------


def test_three_profiles_are_defined() -> None:
    assert set(cli.PROFILES) == {"minimal", "standard", "full"}


def test_minimal_profile_is_redis_plus_qdrant_only() -> None:
    spec = cli.PROFILES["minimal"]
    assert spec.compose_profiles == ["minimal"]
    # Minimal must steer the app at qdrant and switch off the heavy optional tiers,
    # otherwise the agent will try to reach Langfuse/Milvus/Temporal that aren't running.
    assert spec.env["VECTOR_STORE_PROVIDER"] == "qdrant"
    assert spec.env["LANGFUSE_ENABLED"] == "false"
    assert spec.env["TEMPORAL_ENABLED"] == "false"


def test_standard_profile_adds_langfuse_but_not_temporal() -> None:
    spec = cli.PROFILES["standard"]
    assert spec.compose_profiles == ["standard"]
    assert spec.env["LANGFUSE_ENABLED"] == "true"
    assert spec.env["TEMPORAL_ENABLED"] == "false"


def test_full_profile_enables_everything() -> None:
    spec = cli.PROFILES["full"]
    assert spec.compose_profiles == ["full"]
    assert spec.env["LANGFUSE_ENABLED"] == "true"
    assert spec.env["TEMPORAL_ENABLED"] == "true"


# ---------------------------------------------------------------------------
# Bundled compose file
# ---------------------------------------------------------------------------


def test_compose_file_path_points_at_a_real_bundled_file() -> None:
    path = cli.compose_file_path()
    assert path.is_file()
    assert path.name == "docker-compose.yml"


def test_compose_file_is_packaged_under_continuum_infra() -> None:
    path = cli.compose_file_path()
    assert path.parent.name == "infra"
    assert path.parent.parent.name == "continuum"


# ---------------------------------------------------------------------------
# docker compose argv builder
# ---------------------------------------------------------------------------


def test_build_up_command_pins_project_profile_and_detach() -> None:
    argv = cli.build_compose_command("up", "minimal", env_file=Path("/tmp/.env"))
    assert argv[:2] == ["docker", "compose"]
    assert "-f" in argv
    # stable project name so containers are identical regardless of install location
    assert "-p" in argv and "continuum" in argv
    assert "--profile" in argv and "minimal" in argv
    # env-file threads the user's secrets/overrides into compose
    assert "--env-file" in argv and "/tmp/.env" in argv
    assert argv[-2:] == ["up", "-d"]


def test_build_down_command_enables_all_profiles() -> None:
    # docker compose `down` ignores profiled services unless their profile is
    # active, so to tear the WHOLE stack down (whatever tier was started) we must
    # activate every profile — otherwise minimal/standard containers get orphaned.
    argv = cli.build_compose_command("down", env_file=None)
    assert "down" in argv
    for name in ("minimal", "standard", "full"):
        assert name in argv
    assert argv.count("--profile") == 3


def test_build_status_and_logs_enable_all_profiles() -> None:
    # Same reason as down: ps/logs only see services in active profiles.
    for action in ("ps", "logs"):
        argv = cli.build_compose_command(action, env_file=None)
        assert argv.count("--profile") == 3


def test_build_command_omits_env_file_when_absent() -> None:
    argv = cli.build_compose_command("up", "full", env_file=None)
    assert "--env-file" not in argv


# ---------------------------------------------------------------------------
# Managed .env writer
# ---------------------------------------------------------------------------


def test_apply_managed_env_creates_file_with_delimited_block(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    cli.apply_managed_env(env_path, "minimal")

    text = env_path.read_text()
    assert cli.MANAGED_BEGIN in text
    assert cli.MANAGED_END in text
    assert "VECTOR_STORE_PROVIDER=qdrant" in text
    assert "LANGFUSE_ENABLED=false" in text


def test_apply_managed_env_is_idempotent(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    cli.apply_managed_env(env_path, "minimal")
    first = env_path.read_text()
    cli.apply_managed_env(env_path, "minimal")
    second = env_path.read_text()
    assert first == second
    # exactly one managed block, not two stacked copies
    assert second.count(cli.MANAGED_BEGIN) == 1


def test_apply_managed_env_preserves_user_content_outside_block(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=sk-user-secret\n")
    cli.apply_managed_env(env_path, "minimal")
    text = env_path.read_text()
    assert "OPENAI_API_KEY=sk-user-secret" in text
    assert "VECTOR_STORE_PROVIDER=qdrant" in text


def test_apply_managed_env_switching_profiles_replaces_block(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    cli.apply_managed_env(env_path, "minimal")
    cli.apply_managed_env(env_path, "standard")
    text = env_path.read_text()
    assert text.count(cli.MANAGED_BEGIN) == 1
    assert "LANGFUSE_ENABLED=true" in text  # standard
    assert "LANGFUSE_ENABLED=false" not in text  # minimal value gone


def test_apply_managed_env_warns_on_user_value_conflict(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    # User pinned a managed key OUTSIDE the block to a different value.
    env_path.write_text("LANGFUSE_ENABLED=true\n")
    warnings = cli.apply_managed_env(env_path, "minimal")  # minimal wants false
    assert any("LANGFUSE_ENABLED" in w for w in warnings)
