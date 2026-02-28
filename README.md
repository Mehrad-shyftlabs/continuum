# Orchestrator SDK

A Python SDK for agentic AI orchestration with multi-LLM provider support, observability, and error tracking.

## Installation

### Prerequisites

- **Python 3.13** (required)
- pip (latest version recommended)

### Environment Setup

Choose one of the following methods to set up your Python environment:

#### Option 1: Using pyenv (Recommended)

```bash
# Install pyenv (if not already installed)
# macOS: brew install pyenv
# Linux: See https://github.com/pyenv/pyenv#installation

# Install Python 3.13
pyenv install 3.13.9

# Create a virtual environment
pyenv virtualenv 3.13.9 continuum-sdk

# Activate the environment
pyenv activate continuum-sdk
# Or manually:
# source ~/.pyenv/versions/continuum-sdk/bin/activate
```

#### Option 2: Using venv

```bash
# Create a virtual environment
python3.13 -m venv continuum-sdk

# Activate the environment
# macOS/Linux:
source continuum-sdk/bin/activate
# Windows:
# continuum-sdk\Scripts\activate
```

#### Option 3: Using conda

```bash
# Create a conda environment
conda create -n continuum-sdk python=3.13

# Activate the environment
conda activate continuum-sdk
```

### Install the SDK

```bash
# Clone the repository
git clone https://github.com/shyftlabs/continuum.git
cd continuum

# Install in development mode (recommended for development)
pip install -e .

# Or install in production mode
pip install .
```

### Upgrade/Reinstall

If you need to upgrade or force reinstall (e.g., after dependency changes):

```bash
# Upgrade to latest version
pip install -e . --upgrade

# Force reinstall (clears cache and reinstalls all dependencies)
pip install -e . --upgrade --force-reinstall

# Force reinstall without cache (if you encounter dependency issues)
pip install -e . --upgrade --force-reinstall --no-cache-dir
```

### Verify Installation

```bash
# Check SDK version
python -c "from orchestrator import __version__; print(f'SDK version: {__version__}')"

# Check Python version (should be 3.13)
python --version

# Verify core imports work
python -c "from orchestrator.agent import BaseAgent, AgentRunner; from orchestrator.llm import LLMClient; print('✓ All imports successful')"
```

### Troubleshooting

If you encounter deprecation warnings or dependency issues:

```bash
# Upgrade aiohttp explicitly (fixes deprecation warnings in SDK 0.2.0+)
pip install --upgrade "aiohttp>=3.13.2"

# Check installed versions
pip list | grep -E "(aiohttp|shyftlabs-continuum)"

# Reinstall with all dependencies
pip install -e . --upgrade --force-reinstall
```

## Quick Start

```python
from orchestrator.agent import BaseAgent, AgentRunner

# Create an agent
agent = BaseAgent(
    name="my-agent",
    instructions="You are a helpful assistant.",
)

# Run the agent
runner = AgentRunner()
response = await runner.run(
    agent,
    "Hello!",
    user_id="user-123",
)

print(response.content)
```

## Temporal Workflow Integration (Optional)

The SDK supports [Temporal](https://temporal.io) as an optional durable
workflow engine. Any `BaseAgent` can be orchestrated as a fault-tolerant
Temporal workflow with human-in-the-loop approval gates.

### Install

```bash
pip install -e ".[temporal]"
```

### Quick example

```python
from orchestrator.agent import BaseAgent
from orchestrator.temporal import (
    AgentWorkflow,
    WorkflowInput,
    get_agent_registry,
    get_temporal_client,
    get_worker_manager,
)

# Register agents
registry = get_agent_registry()
registry.register(BaseAgent(name="summarizer", instructions="Summarize the input."))
registry.register(BaseAgent(name="reviewer", instructions="Review for accuracy."))

# Connect and start worker
client = get_temporal_client()
await client.connect()
await get_worker_manager().start()

# Run a workflow
handle = await client.start_workflow(
    AgentWorkflow.run,
    WorkflowInput(
        steps=[
            {"type": "agent", "agent_name": "summarizer"},
            {"type": "approval", "description": "Review before publishing"},
            {"type": "agent", "agent_name": "reviewer"},
        ],
        initial_input="Temporal is a durable execution platform...",
    ),
    id="my-workflow",
    task_queue="orchestrator-agents",
)
result = await handle.result()
```

### Features

- **Agent-agnostic**: any `BaseAgent` works as a workflow step
- **Declarative steps**: sequential, parallel, conditional, loop, wait, approval
- **Human-in-the-loop**: approval gates with notifications, escalation, timeout
- **Fault-tolerant**: automatic retries, durable state, workflow cancellation
- **Docker Compose**: Temporal server, UI, and Postgres included

See the [Temporal docs](docs/temporal/) for the full guide:
- [Getting Started](docs/temporal/getting-started.md)
- [Custom Agents](docs/temporal/custom-agents.md)
- [Human-in-the-Loop](docs/temporal/human-in-loop.md)
- [Workflow Patterns](docs/temporal/workflow-patterns.md)
- [Custom Workflows](docs/temporal/custom-workflows.md)
- [Docker Setup](docs/temporal/docker.md)

## Documentation

Full documentation is available in the [docs/](docs/) folder:

- [Installation Guide](docs/installation.md) - Setup and configuration
- [Agent Module](docs/agent.md) - Agent creation and execution
- [LLM Module](docs/llm.md) - Multi-provider LLM client
- [Memory Module](docs/memory.md) - Long-term memory
- [Session Module](docs/session.md) - Conversation history
- [Observability](docs/observability.md) - Tracing and metrics
- [Tools](docs/tools.md) - MCP integration
- [Core](docs/core.md) - Container and lifecycle
- [Temporal Integration](docs/temporal/) - Durable workflow orchestration

## License

MIT
