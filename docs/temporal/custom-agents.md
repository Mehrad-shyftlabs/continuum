# Custom Agents with Temporal

This guide shows how to register your own `BaseAgent` instances with the
Temporal integration and execute them as durable workflow steps.

## How it works

The Temporal integration is **agent-agnostic**. It never imports or references
your agent code directly. Instead:

1. You register `BaseAgent` instances in the **Agent Registry** by name.
2. Temporal activities look up agents by name at runtime and execute them
   through the standard `AgentRunner`.
3. All existing SDK features (LLM, memory, session, tools, observability)
   work unchanged inside Temporal activities.

```
┌─────────────┐       ┌───────────────┐       ┌─────────────────┐
│  Workflow    │──────▶│  Activity     │──────▶│  AgentRunner    │
│  (steps)    │       │  (lookup by   │       │  (your agent)   │
│             │       │   name)       │       │                 │
└─────────────┘       └───────────────┘       └─────────────────┘
                             │
                      ┌──────┴──────┐
                      │ AgentRegistry│
                      └─────────────┘
```

## Defining agents

Create agents exactly as you would without Temporal:

```python
from orchestrator.agent import BaseAgent

research_agent = BaseAgent(
    name="researcher",
    instructions="Research the given topic and return key findings.",
    model="gpt-4o",
    temperature=0.3,
)

writer_agent = BaseAgent(
    name="writer",
    instructions="Write a polished article from the research findings.",
    model="gpt-4o",
    temperature=0.7,
)

editor_agent = BaseAgent(
    name="editor",
    instructions="Edit the article for clarity, grammar, and style.",
    model="gpt-4o-mini",
    temperature=0.2,
)
```

## Registering agents

### Single agent

```python
from orchestrator.temporal import get_agent_registry

registry = get_agent_registry()
registry.register(research_agent)
```

### Multiple agents

```python
registry.register_many([research_agent, writer_agent, editor_agent])
```

### Listing registered agents

```python
print(registry.list_agents())
# ['researcher', 'writer', 'editor']
```

## Running registered agents in workflows

Once registered, reference agents by their `name` string in any workflow input:

```python
from orchestrator.temporal import WorkflowInput, AgentWorkflow, get_temporal_client

client = get_temporal_client()

handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "researcher"},
            {"type": "agent", "agent_name": "writer"},
            {"type": "agent", "agent_name": "editor"},
        ],
        initial_input="Write an article about durable execution patterns.",
    ),
    id="article-pipeline",
    task_queue="orchestrator-agents",
)

result = await handle.result()
print(result.content)  # Final edited article
```

## Customizing per-step behavior

Each `AgentStep` supports per-step overrides:

```python
steps = [
    {
        "type": "agent",
        "agent_name": "researcher",
        "input": "Focus on Temporal and workflow engines",  # override input
        "timeout": 600,    # 10 minutes (default: 300s)
        "retries": 5,      # retry on failure (default: 3)
        "metadata": {"priority": "high"},
    },
    {
        "type": "agent",
        "agent_name": "writer",
        # Uses previous agent's output as input (default behavior)
    },
]
```

| Field | Type | Default | Description |
|---|---|---|---|
| `agent_name` | `str` | required | Name of the registered agent |
| `input` | `str \| None` | `None` | Override input (otherwise uses previous step's output) |
| `timeout` | `int` | `300` | Activity timeout in seconds |
| `retries` | `int` | `3` | Max retry attempts |
| `metadata` | `dict` | `{}` | Passed to the agent as metadata |

## Custom AgentRunner factory

By default, the registry creates a standard `AgentRunner`. If you need custom
configuration, set a runner factory:

```python
from orchestrator.agent import AgentRunner

def my_runner_factory() -> AgentRunner:
    return AgentRunner()  # add custom config as needed

registry.set_runner_factory(my_runner_factory)
```

## Session and user context

Pass `session_id` and `user_id` through the workflow input to maintain
conversation context across agents:

```python
handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "researcher"},
            {"type": "agent", "agent_name": "writer"},
        ],
        initial_input="Research quantum computing breakthroughs.",
        session_id="session-abc123",
        user_id="user-42",
    ),
    id="session-aware-workflow",
    task_queue="orchestrator-agents",
)
```

## Error handling

If an agent raises an exception, the activity catches it and returns an
`AgentActivityResult` with `status="error"` and the error message in the
`error` field. The workflow continues to subsequent steps (the errored agent's
output will be empty).

To trigger a workflow-level failure, the step's retry policy must be exhausted
first (default: 3 attempts).

## Agent not registered?

If a workflow references an agent name that hasn't been registered, the activity
raises `AgentNotRegisteredError` with a helpful message listing available
agents:

```
AgentNotRegisteredError: Agent 'missing-agent' is not registered.
Available agents: ['researcher', 'writer', 'editor']
```

## Next steps

- [Workflow Patterns](workflow-patterns.md) -- sequential, parallel, conditional, loop
- [Human-in-the-Loop](human-in-loop.md) -- approval gates between agent steps
- [Custom Workflows](custom-workflows.md) -- write your own `@workflow.defn`
