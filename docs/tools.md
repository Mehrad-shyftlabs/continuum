# Tools Module

Model Context Protocol (MCP) integration for tool execution.

## Overview

- **ToolExecutor**: Execute MCP tools
- **MCPServer**: Connect to MCP servers (SSE, Stdio, HTTP)
- **Tool Filtering**: Filter tools by name or function
- **Context State**: Maintain tool context across calls

## ToolExecutor

```python
from orchestrator.tools import ToolExecutor, MCPServerStdio

# Create MCP server
server = MCPServerStdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-puppeteer"],
)

# Create executor
executor = ToolExecutor({server: None})
await executor.initialize()

# Get tools
tools = await executor.list_tools()

# Execute tool
result = await executor.execute_tool(
    tool_name="search",
    arguments={"query": "Python SDK"},
)
```

## MCP Server Types

### Stdio Server

```python
from orchestrator.tools import MCPServerStdio, MCPServerStdioParams

server = MCPServerStdio(
    command="python",
    args=["-m", "my_mcp_server"],
    env={"API_KEY": "..."},
)
```

### SSE Server

```python
from orchestrator.tools import MCPServerSse, MCPServerSseParams

server = MCPServerSse(
    url="https://api.example.com/mcp",
    headers={"Authorization": "Bearer ..."},
)
```

### HTTP Server

```python
from orchestrator.tools import MCPServerStreamableHttp, MCPServerStreamableHttpParams

server = MCPServerStreamableHttp(
    url="https://api.example.com/mcp",
    headers={"Authorization": "Bearer ..."},
)
```

## Tool Filtering

Attach the tool filter to the server (not the executor). The executor takes a **tool_registry**: a dict mapping each server to an optional list of allowed tool names (`None` = all tools).

```python
from orchestrator.tools import create_static_tool_filter, MCPServerStreamableHttp, ToolExecutor

# Filter by tool names (use allowed_tool_names, not allowed_tools)
filter_config = create_static_tool_filter(
    allowed_tool_names=["search", "get_weather"],
)

# Attach filter to the server
server = MCPServerStreamableHttp(
    {"url": "https://api.example.com/mcp"},
    tool_filter=filter_config,
)

# Executor uses tool_registry: { server: None } means all tools from that server
executor = ToolExecutor({server: None})
await executor.initialize()
```

## Tool Context State

Maintain context across tool calls:

```python
from orchestrator.tools.types import ToolContextState

# Get context state
context_state = executor.context_state

# Set variable
context_state.set("namespace", "session_id", "session-123")

# Get variable
session_id = context_state.get("namespace", "session_id")
```

## Run Artifacts

Capture MCP tool artifacts (widgets, structured data):

```python
# Artifacts are automatically captured
run_artifacts = executor.run_artifacts

# Get artifacts as dict
artifacts_dict = run_artifacts.to_dict()
```

## Schema Normalization

Normalize tool schemas for LLM compatibility:

```python
from orchestrator.tools import normalize_schema_for_llm

normalized = normalize_schema_for_llm(tool_schema)
```

## Types

- `ToolExecutor`: Main executor class
- `MCPServer`: Base server class
- `ToolContextState`: Context state management
- `RunArtifacts`: Per-run artifacts
- `ToolFilter`: Tool filtering interface

## Adding any MCP (database, API, etc.)

The framework works with **any** MCP server: database MCPs, file servers, custom APIs, etc. Use the transport that fits how the server runs:

| Server type | Transport | Example |
|-------------|-----------|---------|
| Local process (CLI, dev DB server) | **Stdio** | `MCPServerStdio({"command": "npx", "args": ["-y", "@modelcontextprotocol/server-sqlite"]})` |
| Remote / hosted (API, DB gateway) | **Streamable HTTP** | `MCPServerStreamableHttp({"url": "https://mcp.example.com/db"})` |

**Steps to add an MCP and use it everywhere:**

1. **Create the server** (pick one):
   - Stdio (local): `server = MCPServerStdio({"command": "...", "args": [...]})`
   - Remote: `server = MCPServerStreamableHttp({"url": "...", "headers": {...}})`
2. **Connect and build executor:** `await server.connect()` then `executor = ToolExecutor({server: None})` and `await executor.initialize()`.
3. **Get tool definitions for the LLM:** `tool_definitions = await MCPUtil.get_function_tools(server)` then convert to dicts (e.g. `t.to_dict()` or `t.model_dump()`).
4. **Wire to an agent:** pass those dicts as `tools=...` and the same `executor` as `tool_executor` (on the agent or via the runner).
5. **Run:** use `AgentRunner(tool_executor=executor)` or set `agent.tool_executor = executor`; the runner will use it to execute tool calls.

You can attach multiple MCP servers to one executor: `ToolExecutor({server_db: None, server_api: ["search", "get"]})`. Use **tool filtering** (e.g. `create_static_tool_filter(allowed_tool_names=[...])` on the server) to expose only the tools you need.

## Using MCP with agents and across the SDK

- **Single agent:** Create server(s) and executor as above. Build the agent with `tools=<MCP tool dicts>` and either set `agent.tool_executor = executor` or pass `tool_executor=executor` when creating `AgentRunner(...)`. The runner uses that executor to run tool calls and will use the same tools you passed to the agent.
- **Shared executor (multiple agents or entrypoints):** Build one `ToolExecutor` from your MCP server(s), then pass it into `AgentRunner(tool_executor=executor)` or set it on the **Container** with `container.set_tool_executor(executor)` so any runner using that container gets the same MCP tools.
- **Per-agent executor:** Set `agent.tool_executor` on each `BaseAgent`. The runner prefers the agent’s executor when executing that agent’s tool calls.

So yes: adding any MCP (database or otherwise) is the same pattern—create server, connect, build executor, get tools, attach to agent/runner/container—and the same executor is used for execution across the SDK.

## MCP client compatibility

Use the transport that matches your MCP client and deployment:

| Transport | Use case | When to use |
|-----------|-----------|--------------|
| **Stdio** | Local/CLI | Dev, CLI tools, one process per client (e.g. local scripts). |
| **Streamable HTTP** | Remote/web | Preferred for remote servers, Cursor, Claude, web apps; single endpoint, session via `Mcp-Session-Id`, resumable sessions. |
| **SSE** | Legacy | Backward compatibility with existing HTTP+SSE servers only. |

- **Remote servers:** Use `MCPServerStreamableHttp` with `url` and optional `headers` (e.g. `Authorization: Bearer ...`).
- **Session ID:** Streamable HTTP sessions are managed by the MCP SDK; the client sends/receives `Mcp-Session-Id` automatically. There is no API yet to read the session ID from the client; see [MCP Python SDK #942](https://github.com/modelcontextprotocol/python-sdk/issues/942) if you need it for UI/widget correlation.
- **Spec:** [MCP transports](https://modelcontextprotocol.io/docs/concepts/transports).

## Connection and timeouts

- **Validate on connect:** You can pass `validate_on_connect=True` when creating an MCP server (Stdio, SSE, or Streamable HTTP) to call `list_tools()` once after connect and fail fast if the server is misconfigured. Default is `False` so slow servers are not penalized.
- **Timeouts:** Use `client_session_timeout_seconds` for the MCP session read timeout, and for HTTP transports use `timeout` (request) and `sse_read_timeout` (stream). For long-running tool calls, set `sse_read_timeout` to at least 300 (5 minutes). If the SSE connection drops, the SDK can hang; setting these timeouts helps mitigate that.

## Testing

- **Unit tests (no MCP server):** Run `pytest tests/unit/tools/` to test tools util, executor, types, and MCP client logic with mocks. No external MCP server required.
- **Integration tests:** Run `pytest -m integration` to run all integration tests. MCP integration tests can use a mock/in-process server (no network) or a live URL.
- **Live MCP server:** To run tests against a real MCP server, set env `MCP_SERVER_URL` (e.g. `https://your-mcp.example.com/mcp`) and run `pytest tests/integration/test_tools_mcp.py`. Tests marked `live_mcp` require this; CI can run `pytest -m "integration and not live_mcp"` to skip live-only tests when no server is available.
