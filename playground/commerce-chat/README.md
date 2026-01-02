# Petco Retail Agent

An AI-powered shopping assistant for Petco, built with the Orchestrator SDK.

## Features

- 🛒 **Full Shopping Experience**: Browse products, manage cart, checkout
- 🧠 **Personalized Recommendations**: Remembers your pets and preferences
- 💬 **Natural Conversation**: Chat naturally about your pet needs
- 🔧 **Dynamic Tool Usage**: Automatically uses the right tools for each request
- 📊 **Full Observability**: All interactions traced in Langfuse

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Petco Retail Agent                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Single Intelligent Agent                 │   │
│  │                                                           │   │
│  │  • Dynamic tool selection based on user query            │   │
│  │  • No specialist agents (lower latency)                  │   │
│  │  • All decisions made by LLM                             │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│              ┌───────────────┼───────────────┐                  │
│              ▼               ▼               ▼                  │
│       ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│       │  Memory  │    │  Session │    │   MCP    │             │
│       │ (Qdrant) │    │  (Redis) │    │  Tools   │             │
│       └──────────┘    └──────────┘    └──────────┘             │
│              │               │               │                  │
│              └───────────────┼───────────────┘                  │
│                              ▼                                   │
│                      ┌──────────────┐                           │
│                      │   Langfuse   │                           │
│                      │   Tracing    │                           │
│                      └──────────────┘                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Prerequisites

Make sure the following are set up:

```bash
# From project root
cd /path/to/orchestration-layer

# Start Docker services (Redis, Qdrant, Langfuse)
docker-compose up -d

# Set environment variables
export OPENAI_API_KEY=your-api-key

# Set memory isolation mode (for testing mode switching)
# Options: shared, user, agent, run
# Currently testing with "user" mode
export MEMORY_ISOLATION=user
```

### 2. Run the Agent

```bash
# Navigate to playground
cd playground/petco

# Run CLI
poetry run python cli.py
```

### 3. Start Chatting

```
╔═══════════════════════════════════════════════════════════════════╗
║   🐕  Welcome to Petco Shopping Assistant  🐱                     ║
╚═══════════════════════════════════════════════════════════════════╝

You: I have a 2-year old golden retriever named Max. What food do you recommend?

Petco Assistant: I'd be happy to help you find the perfect food for Max! 
For a 2-year old Golden Retriever, I recommend...
[Uses product search tools, provides personalized recommendations]

You: Add the first one to my cart

Petco Assistant: I've added [Product Name] to your cart...
[Uses cart management tools]

You: What's in my cart?

Petco Assistant: Here's what's in your cart...
[Uses cart view tools]
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help message |
| `/tools` | List available MCP tools |
| `/session` | Show session info |
| `/clear` | Clear screen |
| `/quit` | Exit |

## Configuration

Edit `config.py` to customize:

```python
@dataclass
class PetcoConfig:
    # MCP Server
    mcp_url: str = "https://mcp.agentfly.shyftops.io/mcp"
    
    # Agent Settings
    agent_model: str = "gpt-4o"
    max_turns: int = 25
    
    # Memory (personalization)
    enable_memory: bool = True
    
    # Session (conversation history)
    enable_session: bool = True
```

## SDK Features Used

This demo showcases all Orchestrator SDK capabilities using modern integration patterns:

### 1. Container (Dependency Injection)
```python
from orchestrator.core.container import get_container

# Get Container for centralized client management
container = get_container()
llm_client = container.llm_client
memory_client = container.memory_client
session_client = container.session_client
```

### 2. OrchestratorLifecycle
```python
from orchestrator.core.lifecycle import get_lifecycle_manager

# Initialize SDK with health checks and graceful shutdown
lifecycle = get_lifecycle_manager()
await lifecycle.initialize()
# ... use SDK ...
await lifecycle.shutdown()
```

### 3. Agent Orchestration (with Container)
```python
from orchestrator import BaseAgent, AgentRunner
from orchestrator.core.container import get_container

# AgentRunner uses Container by default
runner = AgentRunner(container=get_container())
# Or simply: runner = AgentRunner()  # Uses Container automatically
```

### 4. Memory (Personalization)
```python
# Memory client accessed via Container
memory_client = container.memory_client
# Remembers user preferences, pet info
```

### 5. Session (Conversation History)
```python
# Session client accessed via Container
session_client = container.session_client
# Maintains conversation context
```

### 6. MCP Tools
```python
from orchestrator import MCPServerStreamableHttp, MCPUtil, ToolExecutor
# Dynamic tool discovery and execution
```

### 7. Observability (Automatic)
```python
# Tracing is automatic via decorators
# @trace_agent, @trace_tool, @observe decorators handle all tracing
# No manual tracing code needed!
```

## Available Tools (28)

| Category | Tools |
|----------|-------|
| **Session** | create_session, get_session, touch_session |
| **Products** | list_products, product_without_widgets, get_product |
| **Categories** | list_categories, get_category |
| **Cart** | add_to_cart, bulk_add_to_cart, get_cart, update_cart_item, clear_cart |
| **Orders** | checkout, get_order, list_orders_by_session |
| **Stores** | list_stores, list_stores_without_widget, search_stores_nearby, search_stores_nearby_without_widget, get_store |
| **Appointments** | book_appointment, get_appointment, list_appointments_by_session, list_appointments_by_store, get_available_slots, update_appointment, cancel_appointment |

## Example Interactions

### Product Discovery
```
You: Show me dog food options
You: What cat toys do you have?
You: I need supplies for a new puppy
```

### Cart Management
```
You: Add that to my cart
You: What's in my cart?
You: Clear my cart
```

### Checkout
```
You: I'd like to checkout
You: Show me my orders
```

### Store Finder
```
You: Find stores near San Francisco
You: What stores are available?
```

### Vet Appointments
```
You: I need to book a vet appointment
You: What slots are available tomorrow?
```

### Personalized Help
```
You: I have a 3-year old golden retriever, what food do you recommend?
You: My cat has allergies, what options do you have?
```

## Troubleshooting

### MCP Connection Failed
```
Make sure the MCP server is accessible:
curl https://mcp.agentfly.shyftops.io/mcp
```

### Memory Not Working
```bash
# Check Qdrant is running
docker-compose ps qdrant
curl http://localhost:6333/health
```

### Session Not Working
```bash
# Check Redis is running
docker-compose ps redis-sdk
```

### Missing API Key
```bash
export OPENAI_API_KEY=your-key-here
```

### Memory Isolation Mode
```bash
# Set memory isolation mode (for testing mode switching)
# Options: shared, user, agent, run
# Currently testing with "user" mode
export MEMORY_ISOLATION=user

# The SDK automatically adapts storage and retrieval based on the mode:
# - user: Uses user_id as primary identifier, filters by session_id in metadata for RUN scope
# - run: Uses run_id as primary identifier (no filter needed for RUN scope)
# - agent: Uses agent_id as primary identifier, filters by session_id in metadata for RUN scope
# - shared: Uses agent_id="shared", filters by session_id in metadata for RUN scope
```

## File Structure

```
playground/petco/
├── README.md       # This file
├── config.py       # Configuration
├── agent.py        # Main agent implementation
└── cli.py          # CLI interface
```



uvicorn api:app --host 0.0.0.0 --port 8088 --reload