"""
LLM utility functions for model support checking and validation.

Provides functions to check if models support structured outputs (JSON mode)
and JSON schema validation.
"""

from typing import TYPE_CHECKING

import litellm

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent


def check_response_format_support(model: str, custom_llm_provider: str | None = None) -> bool:
    """
    Check if a model supports response_format parameter.

    This checks for basic JSON mode support (json_object type).

    Args:
        model: Model name (e.g., "gpt-4o", "claude-3-opus")
        custom_llm_provider: Optional custom LLM provider name

    Returns:
        True if the model supports response_format, False otherwise

    Example:
        ```python
        if check_response_format_support("gpt-4o"):
            # Model supports JSON mode
            pass
        ```
    """
    try:
        params = litellm.get_supported_openai_params(model=model, custom_llm_provider=custom_llm_provider)
        return "response_format" in params
    except Exception:
        # If check fails, assume not supported
        return False


def check_json_schema_support(model: str, custom_llm_provider: str | None = None) -> bool:
    """
    Check if a model supports JSON schema (structured outputs).

    This checks for advanced JSON schema support, which allows passing
    Pydantic models or JSON schema dicts with strict validation.

    Args:
        model: Model name (e.g., "gpt-4o-2024-08-06", "gemini-1.5-pro")
        custom_llm_provider: Optional custom LLM provider name

    Returns:
        True if the model supports json_schema, False otherwise

    Example:
        ```python
        if check_json_schema_support("gpt-4o-2024-08-06"):
            # Model supports JSON schema
            pass
        ```
    """
    try:
        return litellm.supports_response_schema(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:
        # If check fails, assume not supported
        return False


def validate_json_schema_config(agent: "BaseAgent") -> tuple[bool, str | None]:
    """
    Validate an agent's JSON schema configuration.

    Checks if the agent's model supports the requested JSON mode configuration
    and validates the configuration is correct.

    Args:
        agent: BaseAgent instance to validate

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if configuration is valid, False otherwise
        - error_message: None if valid, error description if invalid

    Example:
        ```python
        is_valid, error = validate_json_schema_config(agent)
        if not is_valid:
            print(f"Configuration error: {error}")
        ```
    """
    if not agent.enable_json_mode:
        # JSON mode not enabled, nothing to validate
        return True, None

    # Check if model supports response_format
    supports_response_format = check_response_format_support(
        model=agent.model,
        custom_llm_provider=None,  # Could extract from agent if needed
    )

    if not supports_response_format:
        return (
            False,
            f"Model '{agent.model}' does not support response_format. "
            "JSON mode requires a model that supports structured outputs.",
        )

    # If json_schema is provided, check if model supports json_schema
    if agent.json_schema is not None:
        # Check if it's a Pydantic model or a dict
        try:
            from pydantic import BaseModel

            is_pydantic_model = isinstance(agent.json_schema, type) and issubclass(agent.json_schema, BaseModel)
        except Exception:
            is_pydantic_model = False

        if is_pydantic_model or isinstance(agent.json_schema, dict):
            # Check for json_schema support
            supports_json_schema = check_json_schema_support(
                model=agent.model,
                custom_llm_provider=None,  # Could extract from agent if needed
            )

            if not supports_json_schema:
                return (
                    False,
                    f"Model '{agent.model}' does not support JSON schema. "
                    "Use a model that supports structured outputs with schema validation "
                    "(e.g., gpt-4o-2024-08-06, gemini-1.5-pro) or use simple JSON mode "
                    "by setting json_schema=None.",
                )

    return True, None


def supports_tools_with_json_mode(model: str, custom_llm_provider: str | None = None) -> bool:
    """
    Check if a model supports function calling (tools) with JSON mode simultaneously.

    Some models like Gemini don't support using tools/function calling
    together with response_format (JSON mode).

    Args:
        model: Model name (e.g., "gpt-4o", "gemini/gemini-2.5-flash")
        custom_llm_provider: Optional custom LLM provider name

    Returns:
        True if the model supports both tools and JSON mode together, False otherwise

    Example:
        ```python
        if not supports_tools_with_json_mode("gemini/gemini-2.5-flash"):
            # Need to disable JSON mode when tools are present
            pass
        ```
    """
    # Extract provider from model name if not provided
    if custom_llm_provider is None:
        if "/" in model:
            # Model format: provider/model-name
            custom_llm_provider = model.split("/")[0]

    # Models that don't support tools + JSON mode
    unsupported_providers = {
        "gemini",
        "vertex_ai",  # Vertex AI uses Gemini
        "google",  # Google AI Studio
    }

    # Check if provider is in unsupported list
    if custom_llm_provider and custom_llm_provider.lower() in unsupported_providers:
        return False

    # Check model name patterns
    model_lower = model.lower()
    if any(provider in model_lower for provider in ["gemini", "vertex"]):
        return False

    # Default: assume supported (most OpenAI, Anthropic, etc. models support both)
    return True

