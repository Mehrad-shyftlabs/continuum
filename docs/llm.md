# LLM Module

Unified interface for multi-LLM provider support using LiteLLM.

## Overview

- **LLMClient**: Unified client for 100+ LLM providers
- **Structured Outputs**: JSON mode with Pydantic models or JSON schemas
- **Automatic Compatibility**: Handles tools + JSON mode compatibility automatically
- **Context Management**: Automatic context compression and window management
- **Streaming**: Real-time response streaming
- **Tool Calling**: Function/tool calling with tracing
- **Observability**: Full Langfuse integration

## LLMClient

```python
from orchestrator.llm import LLMClient, ChatMessage

client = LLMClient()

# Simple chat
response = await client.chat([
    ChatMessage(role="user", content="Hello!")
])

# With tools
response = await client.chat(
    messages,
    tools=[weather_tool],
    trace_metadata={"task": "weather-lookup"}
)

# Streaming
async for chunk in client.chat_stream(messages):
    print(chunk.content, end="")
```

## Configuration

```python
from orchestrator.llm import LLMConfig

config = LLMConfig(
    model="gpt-4o",
    temperature=0.7,
    max_tokens=4096,
    timeout=60,
)
```

## Structured Outputs (JSON Mode)

The SDK supports LiteLLM's structured output features, allowing you to generate JSON responses with optional schema validation.

### Basic JSON Mode

Simple JSON object mode ensures the LLM returns valid JSON:

```python
from orchestrator.llm import LLMConfig

config = LLMConfig(
    model="gpt-4o-mini",
    json_mode=True,  # Enables {"type": "json_object"}
)

response = await client.chat(messages, config=config)
# response.content will be valid JSON
```

### JSON Schema with Pydantic Models

For structured outputs with validation, use a Pydantic model:

```python
from pydantic import BaseModel
from orchestrator.llm import LLMConfig

class CalendarEvent(BaseModel):
    name: str
    date: str
    participants: list[str]

config = LLMConfig(
    model="gpt-4o-2024-08-06",
    response_format=CalendarEvent,  # Pydantic model
)

response = await client.chat(messages, config=config)
# response.content will be JSON matching CalendarEvent schema
```

### JSON Schema with Raw Dicts

You can also pass a raw JSON schema dictionary:

```python
from orchestrator.llm import LLMConfig

json_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "event_schema",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "date": {"type": "string"},
                "participants": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["name", "date", "participants"],
            "additionalProperties": False
        },
        "strict": True
    }
}

config = LLMConfig(
    model="gpt-4o-2024-08-06",
    response_format=json_schema,
)

response = await client.chat(messages, config=config)
```

### Tools + JSON Mode Compatibility

**Important**: Some models (like Gemini) don't support function calling (tools) with JSON mode simultaneously. The SDK automatically handles this:

- **When tools are present**: The SDK checks if the model supports both tools and JSON mode
- **If unsupported**: JSON mode is automatically disabled to allow tool usage (tools take priority)
- **If supported**: Both tools and JSON mode work together

This happens automatically - no configuration needed:

```python
from orchestrator.llm import LLMClient, LLMConfig
from pydantic import BaseModel

class Result(BaseModel):
    answer: str

# Even if JSON mode is enabled, it will be auto-disabled for Gemini when tools are present
config = LLMConfig(
    model="gemini/gemini-2.5-flash",
    response_format=Result,  # JSON mode enabled
)

# When tools are passed, JSON mode is automatically disabled for incompatible models
response = await client.chat(
    messages,
    config=config,
    tools=[some_tool],  # Tools present
)
# JSON mode was automatically disabled, tools work correctly
```

### Checking Model Support

Before enabling JSON mode, check if your model supports it:

```python
from orchestrator.llm.utils import (
    check_response_format_support,
    check_json_schema_support,
    supports_tools_with_json_mode,
)

# Check basic JSON mode support
if check_response_format_support("gpt-4o"):
    # Model supports json_object mode
    pass

# Check JSON schema support (for structured outputs)
if check_json_schema_support("gpt-4o-2024-08-06"):
    # Model supports json_schema with Pydantic models
    pass

# Check if model supports tools + JSON mode together
if supports_tools_with_json_mode("gemini/gemini-2.5-flash"):
    # Model supports both - JSON mode won't be disabled
    pass
else:
    # Model doesn't support both - JSON mode will be auto-disabled when tools are present
    pass
```

### Supported Models

**JSON Object Mode** (`{"type": "json_object"}`):
- Most OpenAI models (GPT-4, GPT-3.5)
- Most Anthropic models (Claude 3+)
- Most Google models (Gemini)

**JSON Schema Mode** (Pydantic models or `json_schema` dict):
- OpenAI: `gpt-4o-2024-08-06` or later
- Google: Gemini 1.5 Pro, Gemini 1.5 Flash
- Anthropic: Claude 3.5 Sonnet, Claude 3 Opus
- Azure OpenAI: Models with structured outputs enabled
- xAI: Grok-2 or later
- Bedrock: Supported models via AWS Bedrock
- Vertex AI: Gemini and Anthropic models

For a complete list, check LiteLLM's [model support documentation](https://docs.litellm.ai/docs/completion/json_mode).

### Automatic Response Validation

The SDK automatically validates JSON responses when JSON mode is enabled:

- **Logging**: Comprehensive logs show when JSON mode is active and response validation status
- **Format Verification**: Responses are checked to ensure they're valid JSON
- **Schema Validation**: When using Pydantic models, responses are validated against the schema
- **Error Handling**: Graceful degradation if JSON parsing fails (warnings logged, but request doesn't fail)

All validation happens automatically - you'll see logs like:
- `📋 JSON mode active: ...` - When JSON mode is enabled in request
- `✅ LLM response is valid JSON format` - When response is confirmed JSON
- `✅ Successfully parsed JSON response` - When JSON parsing succeeds
- `✅ Successfully validated structured output` - When schema validation succeeds

## Context Management

Automatic context compression when approaching token limits:

```python
from orchestrator.llm import ContextManagementConfig, get_progressive_context_manager

config = ContextManagementConfig(
    enabled=True,
    compression_threshold=0.8,
    summarization_model="gpt-4o-mini",
)

manager = get_progressive_context_manager(config=config)
compressed, result = await manager.compress_if_needed(messages, model="gpt-4o")
```

## LiteLLM Configuration

The SDK uses `litellm_config.yaml` for model pricing and configuration. This file is automatically loaded.

**Location**: The SDK searches for `litellm_config.yaml` in:
1. Current working directory (for development)
2. Project root (if installed from source)
3. Package root (if installed via pip)

**Usage**: Automatically loaded by LLMClient for cost calculation and model configuration.

**Customization**: Set `LITELLM_CONFIG_PATH` environment variable to specify a custom path.

**After pip install**: Place `litellm_config.yaml` in your project root or set `LITELLM_CONFIG_PATH` to point to the file location.

## Supported Providers

- OpenAI (GPT-4, GPT-3.5)
- Google Gemini
- Anthropic Claude
- Azure OpenAI
- 100+ other providers via LiteLLM

## Types

- `ChatMessage`: Message structure
- `LLMResponse`: Response with usage and metadata
- `StreamChunk`: Streaming chunk
- `ToolDefinition`: Tool/function definition

## Exceptions

- `LLMError`: Base LLM error
- `LLMAuthenticationError`: API key issues
- `LLMRateLimitError`: Rate limit exceeded
- `LLMContextLengthError`: Context too long
- `LLMTimeoutError`: Request timeout
