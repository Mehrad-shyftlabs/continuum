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

```python
from orchestrator.tools import create_static_tool_filter

# Filter by tool names
filter_func = create_static_tool_filter(
    allowed_tools=["search", "get_weather"],
)

# Use with executor
executor = ToolExecutor(
    servers={server: None},
    tool_filter=filter_func,
)
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
