# Human-in-the-Loop Approvals

The Temporal integration supports human-in-the-loop (HITL) approval gates
within workflows. A workflow can pause, send a notification to a human
approver, wait for a decision (approve / reject / escalate), and then
continue or abort based on the response.

## Approval step in the generic workflow

Add an `approval` step to any `WorkflowInput`:

```python
from orchestrator.temporal import WorkflowInput, AgentWorkflow, get_temporal_client

client = get_temporal_client()

handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "researcher"},
            {
                "type": "approval",
                "description": "Review research findings before writing",
                "approvers": ["manager@example.com"],
                "timeout": 86400,  # 24 hours
            },
            {"type": "agent", "agent_name": "writer"},
        ],
        initial_input="Analyze market trends for Q4.",
    ),
    id="approval-workflow-1",
    task_queue="orchestrator-agents",
)
```

The workflow will:
1. Run the `researcher` agent.
2. Pause at the approval step and send a notification.
3. Wait up to 24 hours for a human decision.
4. If approved, continue to the `writer` agent.
5. If rejected or timed out, the workflow completes with `status="rejected"` or
   `status="timed_out"`.

## ApprovalStep fields

| Field | Type | Default | Description |
|---|---|---|---|
| `description` | `str` | required | What the approver should review |
| `approvers` | `list[str]` | `[]` | List of approver identifiers |
| `timeout` | `int` | `86400` | Seconds to wait before timing out |
| `auto_approve_if` | `str \| None` | `None` | Reserved for auto-approval conditions |

## Submitting a decision

### Using `HumanInLoopManager` (recommended)

```python
from orchestrator.temporal import HumanInLoopManager, get_temporal_client

client = get_temporal_client()
hitl = HumanInLoopManager(client)

# Approve
await hitl.approve(
    workflow_id="approval-workflow-1",
    request_id="approval-abc123",  # from pending approvals
    decided_by="manager@example.com",
    reason="Looks good, proceed.",
)

# Reject
await hitl.reject(
    workflow_id="approval-workflow-1",
    request_id="approval-abc123",
    decided_by="manager@example.com",
    reason="Needs more data.",
)
```

### Using raw signals

You can also send the approval signal directly:

```python
from orchestrator.temporal import ApprovalDecision
from orchestrator.temporal.workflows.agent_workflow import AgentWorkflow

handle = client.raw_client.get_workflow_handle("approval-workflow-1")
await handle.signal(
    AgentWorkflow.submit_approval,
    ApprovalDecision(
        request_id="approval-abc123",
        decision="approved",
        decided_by="manager@example.com",
        reason="LGTM",
    ),
)
```

## Querying pending approvals

```python
pending = await hitl.get_pending_approvals("approval-workflow-1")
for req in pending:
    print(f"  {req['request_id']}: {req['description']}")
```

## Querying workflow status

```python
status = await hitl.get_workflow_status("approval-workflow-1")
print(status)
# {'status': 'waiting_for_approval', 'current_step_index': 1, ...}
```

## Escalation

If an approval isn't handled in time, you can escalate to another person:

```python
await hitl.escalate(
    workflow_id="approval-workflow-1",
    request_id="approval-abc123",
    escalate_to="director@example.com",
)
```

This sends a new notification (if an escalation handler is configured) and
records an `"escalated"` decision.

## Notification hooks

When an approval step activates, the workflow executes the
`send_notification_activity`. Configure a handler in the registry:

```python
from orchestrator.temporal import get_agent_registry
from orchestrator.temporal.types import NotificationParams

async def slack_notifier(params: NotificationParams) -> None:
    """Send approval request to Slack."""
    import httpx
    async with httpx.AsyncClient() as client:
        await client.post(
            "https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
            json={
                "text": f"Approval needed: {params.payload.get('description')}",
                "workflow_id": params.payload.get("workflow_id"),
            },
        )

registry = get_agent_registry()
registry.set_notification_handler(slack_notifier)
```

### ApprovalNotificationConfig

For more advanced setups, use `ApprovalNotificationConfig`:

```python
from orchestrator.temporal import ApprovalNotificationConfig, HumanInLoopManager

config = ApprovalNotificationConfig(
    handler=slack_notifier,
    timeout_seconds=86400,
    escalation_timeout=7200,
    escalation_handler=escalation_notifier,  # async callable
)

hitl = HumanInLoopManager(client, notification_config=config)
```

## Approval in convenience workflows

### SequentialAgentWorkflow with approval gates

```python
from orchestrator.temporal.workflows.sequential_workflow import (
    SequentialAgentWorkflow,
    SequentialWorkflowInput,
)

handle = await client.start_workflow(
    SequentialAgentWorkflow.run,
    SequentialWorkflowInput(
        agent_names=["researcher", "writer", "editor"],
        initial_input="Write about AI trends.",
        approval_between_steps=True,   # pause between every agent
        approval_timeout=3600,         # 1 hour per gate
    ),
    id="seq-with-approvals",
    task_queue="orchestrator-agents",
)
```

### LoopAgentWorkflow with per-iteration approval

```python
from orchestrator.temporal.workflows.loop_workflow import (
    LoopAgentWorkflow,
    LoopWorkflowInput,
)

handle = await client.start_workflow(
    LoopAgentWorkflow.run,
    LoopWorkflowInput(
        agent_name="refiner",
        initial_input="Draft text...",
        max_iterations=5,
        approval_per_iteration=True,
        approval_timeout=7200,
    ),
    id="loop-with-approvals",
    task_queue="orchestrator-agents",
)
```

## Cancelling a workflow during approval

Send the cancel signal at any time:

```python
handle = await client.get_workflow_handle("approval-workflow-1")
await handle.signal(AgentWorkflow.cancel_workflow)
```

The workflow completes immediately with `status="cancelled"`.

## Timeout behavior

If no decision is received within the approval step's `timeout` seconds,
the workflow:
- Removes the pending approval request.
- Returns `WorkflowResult(status="timed_out")`.

## Next steps

- [Workflow Patterns](workflow-patterns.md) -- all orchestration patterns
- [Custom Workflows](custom-workflows.md) -- build your own approval logic
- [Docker Setup](docker.md) -- infrastructure configuration
