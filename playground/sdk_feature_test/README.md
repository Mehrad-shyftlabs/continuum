# SDK Feature Test

Multi-agent playground that **exercises every major Orchestrator SDK feature** to verify end-to-end behavior.

## Quick run

From repo root (with `src` and `playground` on `PYTHONPATH` or run from repo root):

```bash
# Health checks only (Core: lifecycle, container, health, validate_configuration)
python -m playground.sdk_feature_test run --scenario health

# Workflow only (Core + Router, Sequential, Parallel, Loop; no MCP/memory/session)
python -m playground.sdk_feature_test run --scenario workflow-only

# Full pipeline without Temporal (Core, Session, Memory, MCP, all workflow agents, handoff, structured output, streaming)
python -m playground.sdk_feature_test run --scenario no-temporal

# Full pipeline (same as no-temporal; if TEMPORAL_ENABLED=1 and worker running, runs a Temporal workflow too)
python -m playground.sdk_feature_test run --scenario full
```

Interactive menu (no args):

```bash
python -m playground.sdk_feature_test
```

## What each scenario uses

| Scenario         | Core | Router | Seq/Par/Loop | Handoff | MCP | Memory | Session | Struct out | Stream | Temporal |
|-----------------|------|--------|--------------|---------|-----|--------|---------|------------|--------|----------|
| `health`       | Yes  | No     | No           | No      | No  | No     | No      | No         | No     | No       |
| `workflow-only`| Yes  | Yes    | Yes          | No      | No  | No     | No      | No         | No     | No       |
| `no-temporal`  | Yes  | Yes    | Yes          | Yes     | Yes | Yes    | Yes     | Yes        | Yes    | No       |
| `full`         | Yes  | Yes    | Yes          | Yes     | Yes | Yes    | Yes     | Yes        | Yes    | If enabled |

## Configuration

- **Fake MCP**: By default uses an in-process fake MCP server (echo, add tools). No external server required.
- **Temporal**: Set `TEMPORAL_ENABLED=1` and run a Temporal worker to include the optional workflow step in `full`.
- **Langfuse**: If you see `LANGFUSE_PUBLIC_KEY: Required when LANGFUSE_ENABLED=true`, set `LANGFUSE_ENABLED=false` in your env when not using Langfuse; the pipeline still runs and observability is skipped.
- **Graceful degradation**: If Redis, Qdrant, or Langfuse are unavailable, session/memory/tracing are skipped and the pipeline continues.

### Temporal with Docker

When Temporal is run via the project's `docker-compose.yml` (services `postgres-temporal`, `temporal`, `temporal-ui`), add to your `.env` so the SDK and this playground can connect and run workflows:

```bash
# Enable Temporal in the SDK (required for connection and for "Full with Temporal" to run the workflow step)
TEMPORAL_ENABLED=true
# Host when connecting from your machine to Docker (temporal service exposes 7233)
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
TEMPORAL_TASK_QUEUE=orchestrator-agents
```

- **Temporal server**: `localhost:7233` (container `orchestrator-temporal`, image `temporalio/auto-setup`)
- **Temporal UI**: http://localhost:8233 (container `orchestrator-temporal-ui`)
- **Postgres for Temporal**: `127.0.0.1:5434` (container `orchestrator-postgres-temporal`)

Start the stack: `docker compose up -d postgres-temporal temporal temporal-ui`. For the playground's "Full (with Temporal)" to execute a workflow, a **Temporal worker** must also be running (the worker runs the workflow activities). See [docs/temporal/getting-started.md](../docs/temporal/getting-started.md) and [docs/temporal/docker.md](../docs/temporal/docker.md).

## Features exercised

- **Core**: `initialize_orchestrator`, `get_lifecycle_manager`, `get_container`, `check_all_health`, `validate_configuration`, `shutdown_orchestrator`
- **LLM**: `LLMClient.chat`, `LLMClient.chat_stream`, JSON mode + Pydantic `ReportSummary`
- **Memory**: `MemoryClient.add`, `MemoryClient.search`, scopes
- **Session**: `SessionClient.get_or_create_session`, runner `log_to_session`
- **Tools (MCP)**: `ToolExecutor`, `MCPUtil.get_function_tools`, fake or external server
- **Agents**: `BaseAgent`, `AgentRunner.run` / `run_stream`, `RouterAgent`, `SequentialAgent`, `ParallelAgent`, `LoopAgent`, handoffs
- **Observability**: Tracing via container/Langfuse
- **Temporal** (optional): `AgentRegistry`, `TemporalClient`, `AgentWorkflow`, `HumanInLoopManager` (manual test)
