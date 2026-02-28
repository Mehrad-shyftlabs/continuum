# Custom Workflows

The built-in workflows cover common patterns, but you can write your own
Temporal workflow definitions and still use the SDK's agent activities.

## Writing a custom workflow

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from orchestrator.temporal.types import (
        AgentActivityParams,
        AgentActivityResult,
    )


@dataclass
class MyWorkflowInput:
    topic: str
    user_id: str | None = None


@workflow.defn(sandboxed=False)
class MyCustomWorkflow:
    @workflow.run
    async def run(self, input: MyWorkflowInput) -> dict:
        # Step 1: Run the research agent
        raw = await workflow.execute_activity(
            "run_agent_activity",
            AgentActivityParams(
                agent_name="researcher",
                input=f"Research: {input.topic}",
                user_id=input.user_id,
            ),
            start_to_close_timeout=timedelta(seconds=600),
            retry_policy=RetryPolicy(maximum_attempts=3),
            heartbeat_timeout=timedelta(seconds=60),
            result_type=AgentActivityResult,
        )
        research = (
            raw if isinstance(raw, AgentActivityResult)
            else AgentActivityResult.model_validate(raw)
        )

        if research.status == "error":
            return {"status": "failed", "error": research.error}

        # Step 2: Custom logic (no agent needed)
        word_count = len(research.content.split())

        # Step 3: Run the writer agent only if research is substantial
        if word_count > 50:
            raw = await workflow.execute_activity(
                "run_agent_activity",
                AgentActivityParams(
                    agent_name="writer",
                    input=research.content,
                    user_id=input.user_id,
                ),
                start_to_close_timeout=timedelta(seconds=300),
                retry_policy=RetryPolicy(maximum_attempts=2),
                result_type=AgentActivityResult,
            )
            writer_result = (
                raw if isinstance(raw, AgentActivityResult)
                else AgentActivityResult.model_validate(raw)
            )
            return {
                "status": "completed",
                "content": writer_result.content,
                "word_count": word_count,
            }

        return {
            "status": "completed",
            "content": research.content,
            "word_count": word_count,
            "note": "Research was brief; skipped writer.",
        }
```

### Key conventions

1. **Use `sandboxed=False`** -- the SDK imports (Pydantic models, etc.) don't
   work well inside Temporal's sandbox.
2. **Use dataclasses for workflow input** -- they serialize cleanly with the
   Pydantic data converter.
3. **Reference activities by string name** (`"run_agent_activity"`) -- this is
   the registered activity name.
4. **Set `result_type=AgentActivityResult`** -- ensures proper deserialization.
5. **Import SDK types inside `workflow.unsafe.imports_passed_through()`** --
   this prevents sandbox import restrictions.

## Registering custom workflows

Register your workflow with the `WorkerManager` before starting the worker:

```python
from orchestrator.temporal import get_worker_manager

manager = get_worker_manager()
manager.register_workflow(MyCustomWorkflow)
await manager.start()
```

## Custom activities

You can also define custom activities alongside the built-in ones:

```python
from temporalio import activity


@activity.defn
async def fetch_external_data(url: str) -> str:
    """Custom activity that fetches data from an API."""
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        return resp.text


# Register before starting the worker
manager.register_activity(fetch_external_data)
await manager.start()
```

Then use your custom activity in any workflow:

```python
@workflow.defn(sandboxed=False)
class DataEnrichedWorkflow:
    @workflow.run
    async def run(self, input: MyWorkflowInput) -> dict:
        external = await workflow.execute_activity(
            "fetch_external_data",
            f"https://api.example.com/data/{input.topic}",
            start_to_close_timeout=timedelta(seconds=30),
        )
        # ... use external data with agent activities
```

## Starting your custom workflow

```python
from orchestrator.temporal import get_temporal_client

client = get_temporal_client()
handle = await client.start_workflow(
    MyCustomWorkflow.run,
    MyWorkflowInput(topic="quantum computing", user_id="user-42"),
    id="custom-workflow-1",
    task_queue="orchestrator-agents",
)

result = await handle.result()
print(result)
```

## Adding signals and queries

```python
@workflow.defn(sandboxed=False)
class InteractiveWorkflow:
    def __init__(self):
        self._paused = False
        self._status = "running"

    @workflow.signal
    async def pause(self) -> None:
        self._paused = True

    @workflow.signal
    async def resume(self) -> None:
        self._paused = False

    @workflow.query
    def status(self) -> str:
        return self._status

    @workflow.run
    async def run(self, input: MyWorkflowInput) -> dict:
        for step_name in ["research", "write", "edit"]:
            # Wait until unpaused
            await workflow.wait_condition(lambda: not self._paused)
            self._status = f"running:{step_name}"

            raw = await workflow.execute_activity(
                "run_agent_activity",
                AgentActivityParams(
                    agent_name=step_name,
                    input=input.topic,
                ),
                start_to_close_timeout=timedelta(seconds=300),
                result_type=AgentActivityResult,
            )
            # ... process result

        self._status = "completed"
        return {"status": "completed"}
```

## Testing custom workflows

Use Temporal's time-skipping test environment:

```python
import pytest
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker


@pytest.fixture
async def temporal_env():
    async with await WorkflowEnvironment.start_time_skipping() as env:
        yield env


async def test_my_workflow(temporal_env, mock_registry):
    """Register mock agents, start worker, run workflow."""
    async with Worker(
        temporal_env.client,
        task_queue="test-queue",
        workflows=[MyCustomWorkflow],
        activities=[run_agent_activity, send_notification_activity],
    ):
        handle = await temporal_env.client.start_workflow(
            MyCustomWorkflow.run,
            MyWorkflowInput(topic="test"),
            id="test-custom-1",
            task_queue="test-queue",
        )
        result = await handle.result()
        assert result["status"] == "completed"
```

## Next steps

- [Getting Started](getting-started.md) -- setup and first workflow
- [Workflow Patterns](workflow-patterns.md) -- built-in patterns reference
- [Human-in-the-Loop](human-in-loop.md) -- approval gate patterns
