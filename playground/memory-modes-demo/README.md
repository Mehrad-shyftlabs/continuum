# Memory Modes Demo

Demonstrates the **provider-based memory architecture** with all 4 isolation modes and extensible scope registry.

## Features

- **Provider-Based Architecture**: Uses `Mem0Provider` with mem0's `Memory` client
- **4 Built-in Isolation Modes**: `shared`, `user`, `agent`, `run`
- **Extensible Scope Registry**: Add custom scopes (team, organization, etc.)
- **Direct Memory Operations**: Add, search, list, delete memories directly
- **MemoryScope**: Explicit scope management for memory operations
- **Custom Prompts**: Support for custom fact extraction prompts
- **Both Sync & Async APIs**: Full support for both interfaces

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MemoryModesDemoAgent                      │
│  (Demonstrates memory operations with different modes)       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MemoryClient                            │
│  (High-level API - provider-agnostic thin wrapper)          │
│  • add() / add_sync()      • search() / search_sync()       │
│  • get_all() / get_all_sync()  • delete_all() / ...         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Provider Registry                         │
│  • create_provider("mem0")  • list_providers()              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Mem0Provider                            │
│  (Default provider - implements BaseMemoryProvider)          │
│  • Uses mem0.Memory for all operations                      │
│  • Qdrant vector store for semantic search                  │
│  • LLM-powered fact extraction                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Scope Registry                            │
│  Built-in: shared, user, agent, run                         │
│  Extensible: register_scope("team", "team_id", ...)         │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. **Docker Services Running**:
   ```bash
   # From project root
   docker-compose up -d qdrant redis
   ```

2. **Environment Variables** (in `.env`):
   ```bash
   # Memory Configuration
   MEMORY_ENABLED=true
   MEMORY_PROVIDER=mem0  # Provider selection
   MEMORY_ISOLATION=user  # Options: shared, user, agent, run
   
   # Qdrant
   QDRANT_HOST=localhost
   QDRANT_PORT=6333
   
   # Embedder (OpenAI default)
   EMBEDDER_PROVIDER=openai
   EMBEDDER_MODEL=text-embedding-3-small
   EMBEDDING_DIMS=1536
   
   # Memory LLM (for fact extraction)
   MEMORY_LLM_MODEL=gpt-4o-mini
   ```

## Running the Demo

```bash
cd playground/memory-modes-demo
python cli.py
```

## Commands

### Chat
Just type your message to chat with the agent. It will remember information based on the configured isolation mode.

### Memory Operations
| Command | Description |
|---------|-------------|
| `/add <text>` | Add a memory directly |
| `/search <query>` | Search memories |
| `/list` | List all memories for current scope |
| `/delete-all` | Delete all memories for current scope |

### Context Switching
| Command | Description |
|---------|-------------|
| `/switchuser <id>` | Switch to different user |
| `/switchagent <id>` | Switch to different agent |
| `/newsession` | Start new chat session |

### Info & Control
| Command | Description |
|---------|-------------|
| `/info` | Show memory configuration (provider, isolation mode, etc.) |
| `/help` | Show available commands |
| `/clear` | Clear screen |
| `/quit` | Exit |

## Memory Isolation Modes

### Shared Mode (`MEMORY_ISOLATION=shared`)
- All memories accessible to all users and agents
- No identifier required
- Use for global knowledge base

```bash
export MEMORY_ISOLATION=shared
python cli.py
# Add memory → accessible to everyone
```

### User Mode (`MEMORY_ISOLATION=user`) - Default
- Memories isolated per user
- Requires `user_id` for operations
- Different users have separate memory spaces

```bash
export MEMORY_ISOLATION=user
python cli.py
> My favorite color is blue
> /switchuser bob
> What's my favorite color?  # Won't remember - different user
```

### Agent Mode (`MEMORY_ISOLATION=agent`)
- Memories isolated per agent
- Requires `agent_id` for operations
- Different agents have separate memory spaces

```bash
export MEMORY_ISOLATION=agent
python cli.py
> The project deadline is Friday
> /switchagent sales-bot
> When is the deadline?  # Won't remember - different agent
```

### Run Mode (`MEMORY_ISOLATION=run`)
- Memories isolated per session
- Requires `run_id` for operations
- New sessions have empty memory (ephemeral)

```bash
export MEMORY_ISOLATION=run
python cli.py
> Remember this secret: XYZ123
> /newsession
> What was the secret?  # Won't remember - new session
```

## Testing Isolation Modes

### Test Flow for User Isolation

```bash
export MEMORY_ISOLATION=user
python cli.py

# 1. Start as user "alice"
You: My name is Alice and I love Python
Agent: I'll remember that you're Alice and you love Python.

# 2. Verify memory works for same user
You: What's my name?
Agent: Your name is Alice.

# 3. Switch to user "bob"
You: /switchuser bob
Switched user: alice -> bob

# 4. Test isolation - bob shouldn't know alice's info
You: What's my name?
Agent: I don't have information about your name yet.

# 5. Create memory for bob
You: My name is Bob and I prefer JavaScript
Agent: Got it, Bob! I'll remember you prefer JavaScript.

# 6. Switch back to alice
You: /switchuser alice
Switched user: bob -> alice

# 7. Verify alice's memory is intact
You: What do I love?
Agent: You love Python!
```

### Test Flow for Run/Session Isolation

```bash
export MEMORY_ISOLATION=run
python cli.py

# 1. Add memory in current session
You: The secret code is 12345
Agent: I'll remember the secret code.

# 2. Verify it works
You: What's the secret code?
Agent: The secret code is 12345.

# 3. Start new session
You: /newsession
New session: abc123... -> def456...

# 4. Memory is gone in new session
You: What's the secret code?
Agent: I don't have any secret code stored.
```

## Direct Memory Operations

The demo showcases direct memory API usage:

```python
from orchestrator.memory import MemoryClient, MemoryScope

memory = MemoryClient()

# Add memory with explicit scope
await memory.add(
    "User prefers dark mode",
    user_id="user-123",
    metadata={"category": "preferences"},
)

# Search with custom limit
results = await memory.search(
    "user preferences",
    user_id="user-123",
    limit=10,
)

# Using MemoryScope
scope = MemoryScope.user("user-123")
identifiers = scope.to_identifiers()
# {'user_id': 'user-123'}
```

## Custom Prompts

Add memories with custom fact extraction:

```bash
You: /add My allergies: peanuts, shellfish, dairy (use healthcare prompt)
```

In code:
```python
healthcare_prompt = """
Extract ONLY health-related facts:
- Allergies
- Medications
- Conditions
"""

await memory.add(
    "I'm allergic to peanuts and take metformin",
    user_id="user-123",
    custom_prompt=healthcare_prompt,
)
```

## Adding Custom Scopes

Register a new scope at runtime:

```python
from orchestrator.memory import register_scope, MemoryScope

# Register "team" scope
register_scope(
    name="team",
    required_field="team_id",
    description="Memories isolated per team",
)

# Use it
scope = MemoryScope.from_isolation_mode("team", team_id="engineering")
```

## Configuration Display (`/info`)

```
Memory Configuration:
  Provider: mem0
  Isolation Mode: user
  Is Enabled: True
  User ID: alice
  Agent ID: memory-demo-agent
  Session ID: session-abc123...
  Search Limit: 5
  Embedder: openai / text-embedding-3-small
```

## Files

| File | Description |
|------|-------------|
| `cli.py` | Interactive CLI with commands |
| `agent.py` | Demo agent with memory operations |
| `config.py` | Configuration settings |

## Related Documentation

- [Memory Module Documentation](../../docs/memory.md) - Full API reference
- [Session Management](../../docs/session.md) - Short-term conversation state
- [Agent Documentation](../../docs/agent.md) - Agent integration
