# Temporal Integration -- Getting Started

This guide walks you through installing the Temporal optional dependency,
starting the infrastructure with Docker Compose, registering your agents, and
running your first durable workflow.

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.13 |
| Docker & Docker Compose | latest |
| Orchestrator SDK | installed (`pip install -e .`) |

## 1. Install the Temporal extra

```bash
pip install -e ".[temporal]"
```

This pulls in `temporalio>=1.23.0`. The rest of the SDK continues to work
without it -- Temporal is entirely opt-in.

## 2. Start infrastructure

```bash
docker compose up -d temporal postgres-temporal temporal-ui
```

| Service | URL |
|---|---|
| Temporal gRPC | `localhost:7233` |
| Temporal UI | `http://localhost:8233` |
| Temporal Postgres | `localhost:5434` |

Wait for the health checks to go green:

```bash
docker compose ps          # all services should show "healthy"
```

## 3. Enable Temporal in the SDK

Add the following to your `.env` (or export as environment variables):

```dotenv
TEMPORAL_ENABLED=true
TEMPORAL_HOST=localhost:7233
TEMPORAL_NAMESPACE=default
TEMPORAL_TASK_QUEUE=orchestrator-agents
```

The SDK reads these through the global `Settings` class. You can also override
them programmatically via `TemporalConfig`:

```python
from orchestrator.temporal import TemporalConfig

config = TemporalConfig(
    enabled=True,
    host="localhost:7233",
    namespace="default",
    task_queue="my-queue",
)
```

## 4. Register agents

Any `BaseAgent` can be executed inside a Temporal workflow. Register them with
the global `AgentRegistry`:

```python
from orchestrator.agent import BaseAgent
from orchestrator.temporal import get_agent_registry

summarizer = BaseAgent(
    name="summarizer",
    instructions="Summarize the input concisely.",
)

reviewer = BaseAgent(
    name="reviewer",
    instructions="Review the summary for accuracy.",
)

registry = get_agent_registry()
registry.register_many([summarizer, reviewer])
```

## 5. Connect and start the worker

```python
from orchestrator.temporal import (
    get_temporal_client,
    get_worker_manager,
)

client = get_temporal_client()
await client.connect()

manager = get_worker_manager()
await manager.start()
```

The worker automatically registers all built-in workflows
(`AgentWorkflow`, `SequentialAgentWorkflow`, `ParallelAgentWorkflow`,
`LoopAgentWorkflow`) and activities (`run_agent_activity`,
`send_notification_activity`).

## 6. Run a workflow

### Option A: Generic `AgentWorkflow` (declarative steps)

```python
from orchestrator.temporal import WorkflowInput, AgentWorkflow

handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "summarizer"},
            {"type": "agent", "agent_name": "reviewer"},
        ],
        initial_input="Temporal is an open-source durable execution platform...",
    ),
    id="my-first-workflow",
    task_queue="orchestrator-agents",
)

result = await handle.result()
print(result.status)   # "completed"
print(result.content)  # final output from the last agent
```

### Option B: Convenience `SequentialAgentWorkflow`

```python
from orchestrator.temporal.workflows.sequential_workflow import (
    SequentialAgentWorkflow,
    SequentialWorkflowInput,
)

handle = await client.start_workflow(
    SequentialAgentWorkflow.run,
    SequentialWorkflowInput(
        agent_names=["summarizer", "reviewer"],
        initial_input="Temporal is an open-source durable execution platform...",
    ),
    id="seq-workflow-1",
    task_queue="orchestrator-agents",
)

result = await handle.result()
```

## 7. Verify in the Temporal UI

Open `http://localhost:8233` and navigate to your namespace. You should see the
completed workflow with full execution history.

## 8. Shut down cleanly

```python
await manager.stop()
await client.disconnect()
```

Or stop Docker services:

```bash
docker compose down
```

## Next steps

- [Custom Agents Guide](custom-agents.md) -- registering and running your own agents
- [Human-in-the-Loop](human-in-loop.md) -- approval gates and notification hooks
- [Workflow Patterns](workflow-patterns.md) -- sequential, parallel, conditional, loop
- [Custom Workflows](custom-workflows.md) -- writing your own `@workflow.defn`
- [Docker Setup](docker.md) -- full Docker Compose configuration reference
