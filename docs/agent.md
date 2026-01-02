# Agent Module

The agent module provides the core agent abstraction and execution framework.

## Overview

- **BaseAgent**: Fundamental agent class with tools, handoffs, and memory configuration
- **AgentRunner**: Executes agents with full observability and state management
- **Workflow Agents**: RouterAgent, SequentialAgent, ParallelAgent, LoopAgent
- **Handoff Management**: Agent-to-agent handoffs with history summarization
- **State Persistence**: Redis-based state management

## Core Classes

### BaseAgent

Base class for all agents.

```python
from orchestrator.agent import BaseAgent, AgentMemoryConfig, AgentConfig

agent = BaseAgent(
    name="my-agent",
    instructions="You are a helpful assistant.",
    model="gpt-4o",
    temperature=0.7,
    memory_config=AgentMemoryConfig(
        search_memories=True,
        store_memories=True,
        search_scope=MemoryScope.USER,
    ),
    config=AgentConfig(
        max_turns=25,
        log_to_session=True,
    ),
)
```

**Key Attributes:**
- `name`: Unique agent identifier
- `instructions`: System prompt
- `model`: LLM model to use
- `tools`: List of tool definitions
- `handoffs`: List of agents this agent can hand off to
- `memory_config`: Memory search and storage configuration
- `config`: Execution configuration

### AgentRunner

Executes agents with full observability.

```python
from orchestrator.agent import AgentRunner

runner = AgentRunner()

# Run agent
response = await runner.run(
    agent,
    "Hello!",
    user_id="user-123",
    session_id="session-456",
)

# Stream response
async for event in runner.run_stream(agent, "Tell me a story"):
    if event.type == EventType.CONTENT_DELTA:
        print(event.data["content"], end="")
```

## Workflow Agents

### RouterAgent

Routes requests to appropriate agents based on LLM decision.

```python
from orchestrator.agent import RouterAgent, create_router_agent

router = create_router_agent(
    name="main-router",
    routes=[
        Route(target="support-agent", description="Customer support"),
        Route(target="sales-agent", description="Sales inquiries"),
    ],
)
```

### SequentialAgent

Executes agents in sequence, passing output between them.

```python
from orchestrator.agent import SequentialAgent, create_sequential_agent

pipeline = create_sequential_agent(
    name="processing-pipeline",
    agents=[agent1, agent2, agent3],
)
```

### ParallelAgent

Executes multiple agents in parallel and merges results.

```python
from orchestrator.agent import ParallelAgent, create_parallel_agent

parallel = create_parallel_agent(
    name="parallel-processor",
    agents=[agent1, agent2, agent3],
    merge_strategy=MergeStrategy.LLM_SUMMARIZE,
)
```

### LoopAgent

Executes an agent in a loop until termination condition.

```python
from orchestrator.agent import LoopAgent, create_loop_agent

loop = create_loop_agent(
    name="iterative-agent",
    agent=base_agent,
    max_iterations=10,
)
```

## Handoffs

Agents can hand off to other agents:

```python
from orchestrator.agent import BaseAgent, Handoff

agent = BaseAgent(
    name="support-agent",
    instructions="...",
    handoffs=[
        Handoff(
            target_agent="billing-agent",
            description="Hand off billing inquiries",
        ),
    ],
)
```

## Configuration

### AgentConfig

Controls agent execution behavior:

- `max_turns`: Maximum conversation turns
- `timeout`: Execution timeout in seconds
- `log_to_session`: Whether to save messages to session
- `trace_all_turns`: Whether to trace every turn

### AgentMemoryConfig

Controls memory behavior:

- `search_memories`: Whether to search long-term memory
- `store_memories`: Whether to store new memories
- `search_scope`: Memory scope (USER, AGENT, RUN)
- `store_scope`: Storage scope

## Disabling Memory or Session

```python
# Disable memory
agent = BaseAgent(
    name="stateless-agent",
    memory_config=AgentMemoryConfig(
        search_memories=False,
        store_memories=False,
    ),
)

# Disable session logging
agent = BaseAgent(
    name="no-session-agent",
    config=AgentConfig(
        log_to_session=False,
    ),
)
```

## Structured Outputs

Agents can be configured to generate structured JSON responses with optional schema validation. This is useful when you need consistent, parseable output formats.

### Basic JSON Mode

Enable simple JSON object mode to ensure responses are valid JSON:

```python
from orchestrator.agent import BaseAgent

agent = BaseAgent(
    name="json-agent",
    instructions="You are a helpful assistant that outputs JSON.",
    enable_json_mode=True,  # Enables {"type": "json_object"}
)
```

### JSON Schema with Pydantic Models

For structured outputs with validation, use a Pydantic model:

```python
from pydantic import BaseModel
from orchestrator.agent import BaseAgent

class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    recommendations: list[str]

agent = BaseAgent(
    name="analyst",
    instructions="Analyze data and provide structured results.",
    enable_json_mode=True,
    json_schema=AnalysisResult,  # Pydantic model
    output_schema=AnalysisResult,  # For response parsing
)
```

When the agent runs, the response will:
1. Be validated against the `AnalysisResult` schema by the LLM
2. Be automatically parsed and validated in `AgentResponse.structured_output`

### JSON Schema with Raw Dicts

You can also pass a raw JSON schema dictionary:

```python
from orchestrator.agent import BaseAgent

json_schema_dict = {
    "name": "analysis_schema",
    "schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "confidence": {"type": "number"},
            "recommendations": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["summary", "confidence", "recommendations"],
        "additionalProperties": False
    },
    "strict": True
}

agent = BaseAgent(
    name="analyst",
    instructions="Analyze data and provide structured results.",
    enable_json_mode=True,
    json_schema=json_schema_dict,
    json_strict=True,  # Enforce strict schema validation
)
```

### Configuration Options

- `enable_json_mode: bool = False` - Explicitly enable JSON mode
- `json_schema: dict[str, Any] | type[BaseModel] | None = None` - Optional schema (dict or Pydantic model)
- `json_strict: bool = True` - Whether to use strict mode for JSON schema (default: True)
- `output_schema: type[BaseModel] | None = None` - Pydantic model for parsing/validating responses

### Response Parsing

When `enable_json_mode=True` and `output_schema` is set, the executor automatically:
1. Parses the JSON response content
2. Validates it against the `output_schema` Pydantic model
3. Sets `AgentResponse.structured_output` with the parsed model instance

```python
response = await runner.run(agent, "Analyze this data")

if response.structured_output:
    # response.structured_output is an instance of output_schema
    print(response.structured_output.summary)
    print(response.structured_output.confidence)
else:
    # Fallback to raw content if parsing failed
    print(response.content)
```

### Tools + JSON Mode Compatibility

**Automatic Handling**: The SDK automatically handles compatibility between tools and JSON mode:

- When an agent has both `enable_json_mode=True` and tools (handoffs), the SDK checks model support
- For models that don't support both (like Gemini), JSON mode is automatically disabled when tools are present
- Tools take priority - agent functionality is preserved
- Logs clearly indicate when JSON mode is disabled for compatibility

This is handled transparently - your agent code doesn't need to change:

```python
from orchestrator.agent import BaseAgent
from pydantic import BaseModel

class Result(BaseModel):
    answer: str

# Agent with JSON mode and handoffs (tools)
agent = BaseAgent(
    name="orchestrator",
    enable_json_mode=True,
    json_schema=Result,
    handoffs=[...],  # These are tools
    model="gemini/gemini-2.5-flash",  # Doesn't support tools + JSON
)

# SDK automatically disables JSON mode when handoffs are used
# You'll see a log: "Model doesn't support function calling with JSON mode. Disabling JSON mode to allow tool usage."
```

### Model Support Validation

Before enabling JSON mode, validate your agent's configuration:

```python
from orchestrator.llm.utils import (
    validate_json_schema_config,
    supports_tools_with_json_mode,
)

agent = BaseAgent(
    name="analyst",
    enable_json_mode=True,
    json_schema=AnalysisResult,
    model="gpt-4o-2024-08-06",
)

# Validate configuration
is_valid, error = validate_json_schema_config(agent)
if not is_valid:
    print(f"Configuration error: {error}")

# Check if model supports tools + JSON mode
if agent.handoffs and not supports_tools_with_json_mode(agent.model):
    print("Note: JSON mode will be auto-disabled when handoffs are used")
```

### Example: Complete Agent with Structured Outputs

```python
from pydantic import BaseModel, Field
from orchestrator.agent import BaseAgent, AgentRunner

class TaskResult(BaseModel):
    """Structured output for task completion."""
    completed: bool
    steps: list[str] = Field(..., description="List of steps taken")
    result: str = Field(..., description="Final result")
    errors: list[str] = Field(default_factory=list)

agent = BaseAgent(
    name="task-agent",
    instructions="Complete tasks and report structured results.",
    model="gpt-4o-2024-08-06",  # Model that supports json_schema
    enable_json_mode=True,
    json_schema=TaskResult,
    output_schema=TaskResult,
)

runner = AgentRunner()
response = await runner.run(agent, "Complete this task")

if response.structured_output:
    task_result = response.structured_output
    print(f"Completed: {task_result.completed}")
    print(f"Steps: {task_result.steps}")
    print(f"Result: {task_result.result}")
```

### Automatic Response Validation & Logging

The SDK provides comprehensive logging for JSON mode:

- **Before Request**: Logs when JSON mode is enabled and what schema is used
- **After Response**: Verifies response is valid JSON format
- **Parsing**: Logs successful JSON parsing with extracted keys
- **Validation**: Logs successful schema validation against Pydantic models
- **Errors**: Clear warnings/errors if JSON parsing or validation fails

Example logs you'll see:
```
📋 JSON mode enabled with schema: AnalystOutput for agent analyst
📋 JSON mode active: Pydantic model schema (AnalystOutput) for model gpt-4o
✅ LLM response is valid JSON format (expected with JSON mode)
✅ Successfully parsed JSON response for agent analyst
✅ Successfully validated structured output for agent analyst against AnalystOutput
```

### Supported Models

See the [LLM Module documentation](./llm.md#structured-outputs-json-mode) for a complete list of models that support structured outputs.

**Key Points:**
- Most models support basic `json_object` mode
- Only newer models support `json_schema` with Pydantic models
- Some models (Gemini) don't support tools + JSON mode - SDK handles this automatically
- Always check model support before enabling JSON mode in production

## Types

- `AgentResponse`: Response from agent execution
- `AgentEvent`: Streaming event
- `RunContext`: Execution context
- `RunState`: Persistent state
- `Handoff`: Handoff definition
- `MemoryScope`: Memory isolation scope

## Exceptions

- `AgentError`: Base agent error
- `AgentExecutionError`: Execution failure
- `MaxTurnsExceededError`: Turn limit exceeded
- `HandoffError`: Handoff failure
