# Workflow Patterns

The Temporal integration provides four orchestration patterns out of the box.
Each pattern is available as a step type in the generic `AgentWorkflow` and
as a standalone convenience workflow class.

## 1. Sequential

Agents run one after another. Each agent receives the previous agent's output
as its input.

### Via AgentWorkflow (declarative)

```python
from orchestrator.temporal import WorkflowInput, AgentWorkflow

handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "researcher"},
            {"type": "agent", "agent_name": "writer"},
            {"type": "agent", "agent_name": "editor"},
        ],
        initial_input="Write about distributed systems.",
    ),
    id="sequential-1",
    task_queue="orchestrator-agents",
)
```

### Via SequentialAgentWorkflow (convenience)

```python
from orchestrator.temporal.workflows.sequential_workflow import (
    SequentialAgentWorkflow,
    SequentialWorkflowInput,
)

handle = await client.start_workflow(
    SequentialAgentWorkflow.run,
    SequentialWorkflowInput(
        agent_names=["researcher", "writer", "editor"],
        initial_input="Write about distributed systems.",
    ),
    id="sequential-2",
    task_queue="orchestrator-agents",
)
```

**Options:**
- `approval_between_steps: bool = False` -- insert approval gates between agents
- `approval_timeout: int = 86400` -- timeout per approval gate

## 2. Parallel

Multiple agents run concurrently. Results are merged according to a configurable
strategy.

### Via AgentWorkflow (declarative)

```python
handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {
                "type": "parallel",
                "agents": [
                    {"type": "agent", "agent_name": "analyst-a"},
                    {"type": "agent", "agent_name": "analyst-b"},
                    {"type": "agent", "agent_name": "analyst-c"},
                ],
                "merge_strategy": "concatenate",
            },
        ],
        initial_input="Analyze Q4 market data.",
    ),
    id="parallel-1",
    task_queue="orchestrator-agents",
)
```

### Via ParallelAgentWorkflow (convenience)

```python
from orchestrator.temporal.workflows.parallel_workflow import (
    ParallelAgentWorkflow,
    ParallelWorkflowInput,
)

handle = await client.start_workflow(
    ParallelAgentWorkflow.run,
    ParallelWorkflowInput(
        agent_names=["analyst-a", "analyst-b", "analyst-c"],
        initial_input="Analyze Q4 market data.",
        merge_strategy="structured",
    ),
    id="parallel-2",
    task_queue="orchestrator-agents",
)
```

### Merge strategies

| Strategy | Behavior |
|---|---|
| `concatenate` (default) | Join all outputs with `\n\n` |
| `first_success` | Use the first non-error result |
| `structured` | Dict mapping agent name to output: `{"analyst-a": "...", "analyst-b": "..."}` |

**Options:**
- `timeout_per_agent: int = 300` -- activity timeout for each parallel agent

## 3. Conditional

Run a condition-checking agent. Based on its output, branch into different
step sequences.

```python
handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {
                "type": "conditional",
                "condition_agent": "classifier",
                "if_true": [
                    {"type": "agent", "agent_name": "positive-handler"},
                ],
                "if_false": [
                    {"type": "agent", "agent_name": "negative-handler"},
                ],
            },
        ],
        initial_input="Customer feedback: The product is amazing!",
    ),
    id="conditional-1",
    task_queue="orchestrator-agents",
)
```

The condition agent's output is evaluated as truthy if it contains one of:
`"true"`, `"yes"`, `"1"`, `"approved"`, `"continue"` (case-insensitive).

## 4. Loop

Run a single agent repeatedly until a termination condition is met or the
maximum number of iterations is reached.

### Via AgentWorkflow

Loops are not a first-class step type in the generic workflow, but are available
as a standalone workflow.

### Via LoopAgentWorkflow (convenience)

```python
from orchestrator.temporal.workflows.loop_workflow import (
    LoopAgentWorkflow,
    LoopWorkflowInput,
)

handle = await client.start_workflow(
    LoopAgentWorkflow.run,
    LoopWorkflowInput(
        agent_name="refiner",
        initial_input="Draft: The quick brown fox...",
        max_iterations=5,
        termination_phrase="COMPLETE",
    ),
    id="loop-1",
    task_queue="orchestrator-agents",
)
```

The loop terminates when either:
- The agent's output contains the `termination_phrase` (case-insensitive).
- `max_iterations` is reached.

**Options:**
- `approval_per_iteration: bool = False` -- require approval between iterations
- `approval_timeout: int = 86400` -- timeout per approval

## 5. Wait

Pause the workflow for a fixed duration.

```python
handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "researcher"},
            {"type": "wait", "duration_seconds": 60},  # wait 1 minute
            {"type": "agent", "agent_name": "writer"},
        ],
        initial_input="Topic...",
    ),
    id="wait-1",
    task_queue="orchestrator-agents",
)
```

## Combining patterns

The generic `AgentWorkflow` supports mixing step types in a single workflow:

```python
handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "planner"},
            {
                "type": "approval",
                "description": "Review the plan",
                "timeout": 3600,
            },
            {
                "type": "parallel",
                "agents": [
                    {"type": "agent", "agent_name": "writer-a"},
                    {"type": "agent", "agent_name": "writer-b"},
                ],
                "merge_strategy": "concatenate",
            },
            {
                "type": "conditional",
                "condition_agent": "quality-checker",
                "if_true": [{"type": "agent", "agent_name": "publisher"}],
                "if_false": [{"type": "agent", "agent_name": "editor"}],
            },
        ],
        initial_input="Create a comprehensive guide on Temporal.",
    ),
    id="complex-pipeline",
    task_queue="orchestrator-agents",
)
```

## WorkflowResult

All workflows return a `WorkflowResult`:

```python
result = await handle.result()

result.status              # "completed", "rejected", "timed_out", "failed", "cancelled"
result.content             # final output string
result.step_results        # list[AgentActivityResult] -- per-step results
result.approval_decisions  # list[ApprovalDecision] -- approval audit trail
result.error               # error message if status == "failed"
```

## Signals and queries

All workflows support these interaction points:

| Type | Name | Description |
|---|---|---|
| Signal | `submit_approval` | Submit an approval decision |
| Signal | `cancel_workflow` | Cancel the workflow |
| Signal | `inject_input` | Inject data mid-workflow (AgentWorkflow only) |
| Query | `get_status` | Get current workflow status |
| Query | `get_pending_approvals` | List pending approval requests |

## Next steps

- [Custom Agents](custom-agents.md) -- registering your agents
- [Human-in-the-Loop](human-in-loop.md) -- detailed approval guide
- [Custom Workflows](custom-workflows.md) -- writing your own workflow definitions
