"""
LLM Client - Unified interface for multi-LLM provider support.

Uses LiteLLM to provide access to 100+ LLM providers with a consistent API.
Supports synchronous, asynchronous, and streaming modes.
Includes full observability integration with Langfuse.
"""

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import litellm
from litellm import (
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from orchestrator.config import settings
from orchestrator.llm.callbacks import (
    get_langfuse_metadata,
    setup_langfuse,
)
from orchestrator.llm.config import LLMConfig
from orchestrator.llm.exceptions import (
    LLMAuthenticationError,
    LLMContextLengthError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMServiceUnavailableError,
    LLMTimeoutError,
)
from orchestrator.llm.types import (
    ChatMessage,
    LLMResponse,
    StreamChunk,
    ToolDefinition,
)
from orchestrator.logging import get_logger
from orchestrator.observability.decorators import observe
from orchestrator.observability.trace_context import (
    get_current_session_id,
)

logger = get_logger(__name__)


class LLMClient:
    """
    Unified LLM client with multi-provider support.

    This client uses LiteLLM under the hood to provide access to 100+ LLM providers
    including OpenAI, Anthropic, Google Gemini, Azure OpenAI, and more.

    Features:
        - Synchronous and asynchronous completion methods
        - Streaming support
        - Function/tool calling with tracing
        - Automatic fallback to backup models
        - Retry logic with exponential backoff
        - Langfuse integration for observability
        - Automatic error reporting to Langfuse
        - Trace context propagation

    Example:
        ```python
        from orchestrator.llm import LLMClient, ChatMessage, LLMConfig
        from orchestrator.llm.callbacks import LangfuseTraceContext

        # Initialize client
        client = LLMClient()

        # Simple chat completion (auto-traced)
        response = await client.chat([
            ChatMessage(role="user", content="Hello, how are you?")
        ])
        print(response.content)

        # With explicit trace context
        with LangfuseTraceContext(name="my-agent", user_id="user-123") as trace:
            response = await client.chat([
                ChatMessage(role="user", content="Tell me a story")
            ])
            # LLM call is automatically associated with the trace

        # Function calling with tracing
        response = await client.chat(
            messages,
            tools=[weather_tool],
            trace_metadata={"task": "weather-lookup"}
        )
        ```
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        enable_langfuse: bool = True,
    ):
        """
        Initialize the LLM client.

        Args:
            config: Default configuration for all requests. If not provided,
                   uses global settings from environment variables.
            enable_langfuse: Whether to enable Langfuse logging. Defaults to True.
        """
        self.default_config = config or LLMConfig()
        self._langfuse_enabled = enable_langfuse

        # IMPORTANT: Setup LiteLLM config FIRST (for pricing)
        # Then setup Langfuse callbacks (which need the config for cost calculation)
        self._setup_litellm()

        if enable_langfuse:
            setup_langfuse()

    def _handle_tools_with_json_mode(
        self,
        llm_kwargs: dict[str, Any],
        effective_config: LLMConfig,
        tools_dict: list[dict[str, Any]] | None,
    ) -> None:
        """
        Handle compatibility between tools and JSON mode.

        Some models (like Gemini) don't support function calling with JSON mode.
        This method disables JSON mode when tools are present and the model doesn't support both.

        Args:
            llm_kwargs: Dictionary of kwargs to pass to LiteLLM (modified in place)
            effective_config: LLMConfig instance
            tools_dict: Converted tools dictionary or None
        """
        if tools_dict and (effective_config.json_mode or effective_config.response_format):
            from orchestrator.llm.utils import supports_tools_with_json_mode

            model_supports_both = supports_tools_with_json_mode(
                model=effective_config.model,
                custom_llm_provider=effective_config.custom_llm_provider,
            )

            if not model_supports_both:
                # Disable JSON mode when tools are present (tools take priority)
                if "response_format" in llm_kwargs:
                    logger.warning(
                        f"Model '{effective_config.model}' doesn't support function calling with JSON mode. "
                        "Disabling JSON mode to allow tool usage."
                    )
                    del llm_kwargs["response_format"]

    def _setup_litellm(self) -> None:
        """Configure LiteLLM settings."""
        # Set verbosity
        litellm.set_verbose = settings.litellm_verbose

        # Configure API keys from environment
        if settings.openai_api_key:
            litellm.openai_key = settings.openai_api_key

        # Load YAML config file
        # Default to litellm_config.yaml in project root if not specified
        # Try multiple locations: current working directory, package root, or explicit path
        config_path = settings.litellm_config_path or "litellm_config.yaml"

        if os.path.isabs(config_path):
            config_path = Path(config_path)
        else:
            # Try current working directory first (for development)
            cwd_path = Path.cwd() / config_path
            if cwd_path.exists():
                config_path = cwd_path
            else:
                # Try package root (for installed package)
                # Go up from src/orchestrator/llm/client.py to project root
                package_root = Path(__file__).parent.parent.parent.parent
                package_path = package_root / config_path
                if package_path.exists():
                    config_path = package_path
                else:
                    # Fallback to current working directory
                    config_path = Path.cwd() / config_path

        if config_path.exists():
            # Load LiteLLM config from YAML file
            # This loads models, pricing, and other settings
            # LiteLLM uses this for automatic cost calculation in Langfuse callbacks
            # Set config_path - LiteLLM will load it automatically
            litellm.config_path = str(config_path)

            # Try to explicitly load config if method is available
            if hasattr(litellm, "load_config"):
                try:
                    litellm.load_config(config_path=str(config_path))
                except Exception as e:
                    # load_config might not be available or might fail
                    logger.debug(f"load_config not available or failed: {e}")

            logger.info(
                f"Loaded LiteLLM config from: {config_path}. "
                "Pricing will be used for automatic cost calculation in Langfuse."
            )
        else:
            logger.warning(
                f"LiteLLM config file not found: {config_path}. "
                "Pricing will use LiteLLM defaults. Cost tracking may not work properly."
            )

        # Note: Other API keys are automatically picked up by LiteLLM
        # from standard environment variables

    def _convert_messages(
        self, messages: list[ChatMessage] | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert messages to LiteLLM format."""
        result = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                result.append(msg.to_dict())
            else:
                result.append(msg)
        return result

    def _convert_tools(
        self, tools: list[ToolDefinition] | list[dict[str, Any]] | None
    ) -> list[dict[str, Any]] | None:
        """Convert tools to LiteLLM format."""
        if tools is None:
            return None
        result = []
        for tool in tools:
            if isinstance(tool, ToolDefinition):
                result.append(tool.to_dict())
            else:
                result.append(tool)
        return result

    def _build_metadata(
        self,
        tools: list[dict[str, Any]] | None = None,
        trace_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Build metadata for Langfuse tracing.

        Uses contextvars from @observe decorator for trace context.
        """
        if not self._langfuse_enabled:
            return trace_metadata or {}

        # Get metadata with trace context from contextvars
        metadata = get_langfuse_metadata(
            custom_metadata=trace_metadata,
        )

        # Add tool information for tracing
        if tools:
            tool_names = [t.get("function", {}).get("name", "unknown") for t in tools]
            metadata["tools_available"] = tool_names

        return metadata

    def _handle_exception(
        self,
        e: Exception,
        model: str,
    ) -> None:
        """
        Convert LiteLLM exceptions to custom exceptions.

        Note: Error reporting is handled automatically by @observe decorator.
        """
        provider = self._get_provider_from_model(model)

        # Build context for error
        error_context = {
            "model": model,
            "provider": provider,
        }

        if isinstance(e, AuthenticationError):
            raise LLMAuthenticationError(
                message=str(e),
                model=model,
                provider=provider,
                original_error=e,
                context=error_context,
            ) from e
        elif isinstance(e, RateLimitError):
            raise LLMRateLimitError(
                message=str(e),
                model=model,
                provider=provider,
                original_error=e,
                context=error_context,
            ) from e
        elif isinstance(e, Timeout):
            raise LLMTimeoutError(
                message=str(e),
                model=model,
                provider=provider,
                original_error=e,
                context=error_context,
            ) from e
        elif isinstance(e, ContextWindowExceededError):
            raise LLMContextLengthError(
                message=str(e),
                model=model,
                provider=provider,
                original_error=e,
                context=error_context,
            ) from e
        elif isinstance(e, BadRequestError):
            raise LLMInvalidRequestError(
                message=str(e),
                model=model,
                provider=provider,
                original_error=e,
                context=error_context,
            ) from e
        elif isinstance(e, ServiceUnavailableError):
            raise LLMServiceUnavailableError(
                message=str(e),
                model=model,
                provider=provider,
                original_error=e,
                context=error_context,
            ) from e
        else:
            raise LLMError(
                message=str(e),
                model=model,
                provider=provider,
                original_error=e,
                context=error_context,
            ) from e

    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider name from model string."""
        if "/" in model:
            return model.split("/")[0]
        if model.startswith("gpt"):
            return "openai"
        if model.startswith("claude"):
            return "anthropic"
        if model.startswith("gemini"):
            return "google"
        return "unknown"

    # =========================================================================
    # Synchronous Methods
    # =========================================================================

    @observe(name="llm_chat_sync", capture_output=True)
    def chat_sync(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Synchronous chat completion.

        All calls are automatically traced via @observe decorator.
        Trace context is automatically captured from contextvars.

        Args:
            messages: List of chat messages.
            config: Optional config overrides.
            tools: Optional list of tool definitions for function calling.
            tool_choice: Optional tool choice specification.
            trace_metadata: Additional metadata for tracing (merged with contextvars).
            **kwargs: Additional arguments passed to LiteLLM.

        Returns:
            LLMResponse with the completion result.

        Example:
            ```python
            response = client.chat_sync([
                ChatMessage(role="user", content="Hello")
            ])
            ```
        """
        effective_config = config or self.default_config
        llm_kwargs = effective_config.to_litellm_kwargs()
        llm_kwargs.update(kwargs)

        messages_dict = self._convert_messages(messages)
        tools_dict = self._convert_tools(tools)

        # Handle tools + JSON mode compatibility
        self._handle_tools_with_json_mode(llm_kwargs, effective_config, tools_dict)

        # Log JSON mode status in request
        if "response_format" in llm_kwargs:
            response_format = llm_kwargs["response_format"]
            if isinstance(response_format, dict):
                if response_format.get("type") == "json_object":
                    logger.info(
                        f"📋 JSON mode active: json_object mode for model {effective_config.model}",
                        extra={"model": effective_config.model, "json_mode": "json_object"},
                    )
                elif response_format.get("type") == "json_schema":
                    logger.info(
                        f"📋 JSON mode active: json_schema mode for model {effective_config.model}",
                        extra={
                            "model": effective_config.model,
                            "json_mode": "json_schema",
                            "strict": response_format.get("json_schema", {}).get("strict", False),
                        },
                    )
            elif isinstance(response_format, type):
                logger.info(
                    f"📋 JSON mode active: Pydantic model schema ({response_format.__name__}) for model {effective_config.model}",
                    extra={
                        "model": effective_config.model,
                        "json_mode": "pydantic_schema",
                        "schema_name": response_format.__name__,
                    },
                )

        if tools_dict:
            llm_kwargs["tools"] = tools_dict
        if tool_choice:
            llm_kwargs["tool_choice"] = tool_choice

        # Build tracing metadata (uses contextvars from @observe)
        metadata = self._build_metadata(
            tools=tools_dict,
            trace_metadata=trace_metadata,
        )
        if metadata:
            llm_kwargs["metadata"] = metadata

        try:
            logger.debug(f"Attempting sync completion with model: {effective_config.model}")
            response = litellm.completion(messages=messages_dict, **llm_kwargs)
            llm_response = LLMResponse.from_litellm_response(response)

            # Log response format verification if JSON mode was expected
            if "response_format" in llm_kwargs and llm_response.content:
                import json

                try:
                    content_stripped = llm_response.content.strip()
                    is_json = (
                        content_stripped.startswith("{") and content_stripped.endswith("}")
                    ) or (content_stripped.startswith("[") and content_stripped.endswith("]"))
                    if is_json:
                        parsed = json.loads(llm_response.content)
                        logger.info(
                            f"✅ LLM response is valid JSON format (expected with JSON mode)",
                            extra={
                                "model": effective_config.model,
                                "json_keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                                "content_length": len(llm_response.content),
                            },
                        )
                    else:
                        logger.warning(
                            f"⚠️ LLM response doesn't appear to be JSON format despite JSON mode being enabled",
                            extra={
                                "model": effective_config.model,
                                "content_preview": content_stripped[:100],
                            },
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"⚠️ LLM response is not valid JSON despite JSON mode being enabled",
                        extra={
                            "model": effective_config.model,
                            "content_preview": llm_response.content[:100] if llm_response.content else None,
                        },
                    )

            return llm_response
        except Exception as e:
            # LiteLLM handles fallbacks automatically if configured
            # If all fallbacks exhausted, it will raise an exception
            logger.warning(f"Completion failed: {e}")
            self._handle_exception(e, effective_config.model)
            # Should not reach here, but just in case
            raise

    @observe(name="llm_chat_stream_sync", capture_output=False)
    def chat_stream_sync(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """
        Synchronous streaming chat completion.

        All calls are automatically traced via @observe decorator.
        Trace context is automatically captured from contextvars.

        Args:
            messages: List of chat messages.
            config: Optional config overrides.
            tools: Optional list of tool definitions.
            tool_choice: Optional tool choice specification.
            trace_metadata: Additional metadata for tracing (merged with contextvars).
            **kwargs: Additional arguments passed to LiteLLM.

        Yields:
            StreamChunk objects as they arrive.

        Example:
            ```python
            for chunk in client.chat_stream_sync([
                ChatMessage(role="user", content="Tell me a story")
            ]):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            ```
        """
        effective_config = config or self.default_config
        llm_kwargs = effective_config.to_litellm_kwargs()
        llm_kwargs.update(kwargs)
        llm_kwargs["stream"] = True

        messages_dict = self._convert_messages(messages)
        tools_dict = self._convert_tools(tools)

        # Handle tools + JSON mode compatibility
        self._handle_tools_with_json_mode(llm_kwargs, effective_config, tools_dict)

        if tools_dict:
            llm_kwargs["tools"] = tools_dict
        if tool_choice:
            llm_kwargs["tool_choice"] = tool_choice

        # Build tracing metadata (uses contextvars from @observe)
        metadata = self._build_metadata(
            tools=tools_dict,
            trace_metadata=trace_metadata,
        )
        if metadata:
            llm_kwargs["metadata"] = metadata

        try:
            logger.debug(f"Starting sync stream with model: {effective_config.model}")
            response = litellm.completion(messages=messages_dict, **llm_kwargs)

            for chunk in response:
                yield StreamChunk.from_litellm_chunk(chunk)

        except Exception as e:
            logger.warning(f"Streaming failed: {e}")
            self._handle_exception(e, effective_config.model)
            # Should not reach here, but just in case
            raise

    # =========================================================================
    # Asynchronous Methods
    # =========================================================================

    @observe(name="llm_chat", capture_output=True)
    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        trace_metadata: dict[str, Any] | None = None,
        # Session management
        auto_session: bool = True,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Asynchronous chat completion.

        This is the primary method for making LLM calls in async contexts.
        All calls are automatically traced via @observe decorator.
        Trace context is automatically captured from contextvars.

        Args:
            messages: List of chat messages.
            config: Optional config overrides.
            tools: Optional list of tool definitions for function calling.
            tool_choice: Optional tool choice specification.
            session_id: Session ID for conversation grouping (also from contextvars if not provided).
            trace_metadata: Additional metadata for tracing (merged with contextvars).
            auto_session: Whether to automatically load/save from session.
                         Set to False when caller manages the message loop
                         (e.g., AgentRunner). Defaults to True.
            **kwargs: Additional arguments passed to LiteLLM.

        Returns:
            LLMResponse with the completion result.

        Example:
            ```python
            # Basic usage (auto-traced)
            response = await client.chat([
                ChatMessage(role="system", content="You are a helpful assistant."),
                ChatMessage(role="user", content="What is Python?"),
            ])
            print(response.content)

            # Function calling
            response = await client.chat(
                messages,
                tools=[weather_tool],
                trace_metadata={"task": "tool-use"}
            )
            if response.tool_calls:
                for call in response.tool_calls:
                    print(f"Tool: {call.function.name}")
                    print(f"Args: {call.function.arguments}")

            # When caller manages message loop (disable auto-session)
            response = await client.chat(
                messages,
                session_id="session-123",
                auto_session=False,  # Don't auto-load/save
            )
            ```
        """
        effective_config = config or self.default_config
        llm_kwargs = effective_config.to_litellm_kwargs()
        llm_kwargs.update(kwargs)

        # Use provided session_id or get from contextvars
        # Trace context is automatically available via contextvars (set by @observe decorator)
        effective_session_id = session_id or get_current_session_id()

        # Auto-load conversation history from session if session_id is provided and auto_session is enabled
        # NOTE: When auto_session=False, the caller is managing the message loop (e.g., AgentRunner)
        # and will handle session loading/saving separately.
        if effective_session_id and auto_session:
            try:
                from orchestrator.core.container import get_container

                session_client = get_container().session_client
                if session_client.is_enabled:
                    # Get conversation history from session
                    # Session client will get trace_id/span_id from contextvars via @observe
                    history_messages = await session_client.get_conversation_history(
                        session_id=effective_session_id,
                    )

                    if history_messages:
                        # Prepend history to provided messages
                        # Convert history to dict format if needed
                        history_dicts = [
                            msg.to_dict() if hasattr(msg, "to_dict") else msg
                            for msg in history_messages
                        ]
                        # Combine: history + new messages
                        messages_dict = history_dicts + self._convert_messages(messages)
                        logger.debug(
                            f"Loaded {len(history_messages)} messages from session: {effective_session_id}",
                            extra={"total_messages": len(messages_dict)},
                        )
                    else:
                        messages_dict = self._convert_messages(messages)
                else:
                    messages_dict = self._convert_messages(messages)
            except Exception as e:
                # If session loading fails, continue with provided messages
                logger.warning(
                    f"Failed to load session history: {e}, continuing with provided messages"
                )
                messages_dict = self._convert_messages(messages)
        else:
            messages_dict = self._convert_messages(messages)

        # Apply context management (compression/summarization) if enabled
        try:
            from orchestrator.llm.context_management import (
                get_progressive_context_manager,
            )

            context_manager = get_progressive_context_manager()
            if context_manager.config.enabled:
                # Context manager will get trace_id/span_id from contextvars automatically
                messages_dict, compression_result = await context_manager.compress_if_needed(
                    messages=messages_dict,
                    model=effective_config.model,
                )

                if compression_result.was_compressed:
                    logger.info(
                        f"Context compressed: {compression_result.original_token_count} → "
                        f"{compression_result.compressed_token_count} tokens "
                        f"({compression_result.compression_ratio:.1%} ratio)"
                    )
        except Exception as e:
            # Context management failure should not break the request
            logger.warning(f"Context management failed, continuing without compression: {e}")

        tools_dict = self._convert_tools(tools)

        # Handle tools + JSON mode compatibility
        self._handle_tools_with_json_mode(llm_kwargs, effective_config, tools_dict)

        # Log JSON mode status in request
        if "response_format" in llm_kwargs:
            response_format = llm_kwargs["response_format"]
            if isinstance(response_format, dict):
                if response_format.get("type") == "json_object":
                    logger.info(
                        f"📋 JSON mode active: json_object mode for model {effective_config.model}",
                        extra={"model": effective_config.model, "json_mode": "json_object"},
                    )
                elif response_format.get("type") == "json_schema":
                    logger.info(
                        f"📋 JSON mode active: json_schema mode for model {effective_config.model}",
                        extra={
                            "model": effective_config.model,
                            "json_mode": "json_schema",
                            "strict": response_format.get("json_schema", {}).get("strict", False),
                        },
                    )
            elif isinstance(response_format, type):
                logger.info(
                    f"📋 JSON mode active: Pydantic model schema ({response_format.__name__}) for model {effective_config.model}",
                    extra={
                        "model": effective_config.model,
                        "json_mode": "pydantic_schema",
                        "schema_name": response_format.__name__,
                    },
                )

        if tools_dict:
            llm_kwargs["tools"] = tools_dict
        if tool_choice:
            llm_kwargs["tool_choice"] = tool_choice

        # Build tracing metadata (uses contextvars from @observe)
        metadata = self._build_metadata(
            tools=tools_dict,
            trace_metadata=trace_metadata,
        )
        if metadata:
            llm_kwargs["metadata"] = metadata

        try:
            logger.debug(f"Attempting async completion with model: {effective_config.model}")
            # LiteLLM handles fallbacks automatically if configured via fallbacks parameter
            response = await litellm.acompletion(messages=messages_dict, **llm_kwargs)
            llm_response = LLMResponse.from_litellm_response(response)

            # Log response format verification if JSON mode was expected
            if "response_format" in llm_kwargs and llm_response.content:
                import json

                try:
                    content_stripped = llm_response.content.strip()
                    is_json = (
                        content_stripped.startswith("{") and content_stripped.endswith("}")
                    ) or (content_stripped.startswith("[") and content_stripped.endswith("]"))
                    if is_json:
                        parsed = json.loads(llm_response.content)
                        logger.info(
                            f"✅ LLM response is valid JSON format (expected with JSON mode)",
                            extra={
                                "model": effective_config.model,
                                "json_keys": list(parsed.keys()) if isinstance(parsed, dict) else None,
                                "content_length": len(llm_response.content),
                            },
                        )
                    else:
                        logger.warning(
                            f"⚠️ LLM response doesn't appear to be JSON format despite JSON mode being enabled",
                            extra={
                                "model": effective_config.model,
                                "content_preview": content_stripped[:100],
                            },
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"⚠️ LLM response is not valid JSON despite JSON mode being enabled",
                        extra={
                            "model": effective_config.model,
                            "content_preview": llm_response.content[:100] if llm_response.content else None,
                        },
                    )

            # Auto-save messages to session if session_id is provided and auto_session is enabled
            # NOTE: When auto_session=False, the caller is managing the message loop
            if effective_session_id and auto_session:
                try:
                    from orchestrator.core.container import get_container

                    session_client = get_container().session_client
                    if session_client.is_enabled:
                        # Save new user messages (those not in history)
                        # Convert provided messages to ChatMessage if needed
                        new_messages = self._convert_messages(messages)
                        for msg_dict in new_messages:
                            if isinstance(msg_dict, dict):
                                msg_role = msg_dict.get("role")
                                msg_content = msg_dict.get("content")
                                # Only save user messages with content (assistant will be saved after)
                                # Skip tool messages and messages without content for memory
                                if msg_role == "user" and msg_content:
                                    from orchestrator.llm.types import ChatMessage

                                    user_msg = ChatMessage(**msg_dict)
                                    # Session client will get trace_id/span_id from contextvars
                                    await session_client.add_message(
                                        session_id=effective_session_id,
                                        message=user_msg,
                                        store_in_memory=True,  # Also store in long-term memory
                                    )

                        # Create assistant message from response
                        from orchestrator.llm.types import ChatMessage

                        assistant_message = ChatMessage(
                            role="assistant",
                            content=llm_response.content,
                            tool_calls=llm_response.tool_calls,
                            function_call=llm_response.function_call,
                        )

                        # Add to session - only store in memory if it has content and no tool_calls
                        # Tool call responses are intermediate and not useful for long-term memory
                        should_store_in_memory = (
                            llm_response.content is not None
                            and llm_response.content.strip()
                            and not llm_response.tool_calls
                        )

                        # Session client will get trace_id/span_id from contextvars
                        await session_client.add_message(
                            session_id=effective_session_id,
                            message=assistant_message,
                            store_in_memory=should_store_in_memory,
                        )

                        logger.debug(f"Saved messages to session: {effective_session_id}")
                except Exception as e:
                    # If session saving fails, log but don't fail the request
                    logger.warning(f"Failed to save messages to session: {e}")

            return llm_response

        except Exception as e:
            # LiteLLM handles fallbacks automatically if configured
            # If all fallbacks exhausted, it will raise an exception
            logger.warning(f"Completion failed: {e}")
            self._handle_exception(e, effective_config.model)
            # Should not reach here, but just in case
            raise

    @observe(name="llm_chat_stream", capture_output=False)
    async def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        config: LLMConfig | None = None,
        tools: list[ToolDefinition] | list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        *,
        trace_metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """
        Asynchronous streaming chat completion.

        All calls are automatically traced via @observe decorator.
        Trace context is automatically captured from contextvars.

        Args:
            messages: List of chat messages.
            config: Optional config overrides.
            tools: Optional list of tool definitions.
            tool_choice: Optional tool choice specification.
            trace_metadata: Additional metadata for tracing (merged with contextvars).
            **kwargs: Additional arguments passed to LiteLLM.

        Yields:
            StreamChunk objects as they arrive.

        Example:
            ```python
            async for chunk in client.chat_stream([
                ChatMessage(role="user", content="Tell me a story")
            ]):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
            ```
        """
        effective_config = config or self.default_config
        llm_kwargs = effective_config.to_litellm_kwargs()
        llm_kwargs.update(kwargs)
        llm_kwargs["stream"] = True

        messages_dict = self._convert_messages(messages)

        # Apply context management (compression/summarization) if enabled
        # Compress before streaming starts
        try:
            from orchestrator.llm.context_management import (
                get_progressive_context_manager,
            )

            context_manager = get_progressive_context_manager()
            if context_manager.config.enabled:
                # Context manager will get trace_id/span_id from contextvars
                messages_dict, compression_result = await context_manager.compress_if_needed(
                    messages=messages_dict,
                    model=effective_config.model,
                )

                if compression_result.was_compressed:
                    logger.info(
                        f"Context compressed before streaming: {compression_result.original_token_count} → "
                        f"{compression_result.compressed_token_count} tokens"
                    )
        except Exception as e:
            # Context management failure should not break the request
            logger.warning(f"Context management failed, continuing without compression: {e}")

        tools_dict = self._convert_tools(tools)

        # Handle tools + JSON mode compatibility
        self._handle_tools_with_json_mode(llm_kwargs, effective_config, tools_dict)

        if tools_dict:
            llm_kwargs["tools"] = tools_dict
        if tool_choice:
            llm_kwargs["tool_choice"] = tool_choice

        # Build tracing metadata (uses contextvars from @observe)
        metadata = self._build_metadata(
            tools=tools_dict,
            trace_metadata=trace_metadata,
        )
        if metadata:
            llm_kwargs["metadata"] = metadata

        try:
            logger.debug(f"Starting async stream with model: {effective_config.model}")
            response = await litellm.acompletion(messages=messages_dict, **llm_kwargs)

            async for chunk in response:
                yield StreamChunk.from_litellm_chunk(chunk)

        except Exception as e:
            logger.warning(f"Streaming failed: {e}")
            self._handle_exception(e, effective_config.model)
            # Should not reach here, but just in case
            raise

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @observe(name="llm_get_model_info", capture_output=True)
    def get_model_info(self, model: str | None = None) -> dict[str, Any]:
        """
        Get information about a model.

        Args:
            model: Model name. If not provided, uses the default model.

        Returns:
            Dictionary with model information including max tokens, cost, etc.
        """
        model = model or self.default_config.model
        try:
            return litellm.get_model_info(model)
        except Exception as e:
            logger.warning(f"Could not get model info for {model}: {e}")
            return {}

    def get_supported_models(self) -> list[str]:
        """
        Get list of all supported models.

        Returns:
            List of model names supported by LiteLLM.
        """
        return list(litellm.model_list)

    @observe(name="llm_count_tokens", capture_output=True)
    def count_tokens(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        model: str | None = None,
    ) -> int:
        """
        Count tokens for a list of messages.

        Args:
            messages: List of chat messages.
            model: Model to count tokens for. Uses default if not provided.

        Returns:
            Number of tokens.
        """
        model = model or self.default_config.model
        messages_dict = self._convert_messages(messages)
        try:
            return litellm.token_counter(model=model, messages=messages_dict)
        except Exception as e:
            logger.warning(f"Could not count tokens: {e}")
            return 0

    def get_max_tokens(self, model: str | None = None) -> int | None:
        """
        Get maximum context length for a model.

        Args:
            model: Model name. Uses default if not provided.

        Returns:
            Maximum number of tokens, or None if unknown.
        """
        model = model or self.default_config.model
        try:
            return litellm.get_max_tokens(model)
        except Exception:
            return None

    @observe(name="llm_check_health", capture_output=True)
    async def check_health(self, model: str | None = None) -> bool:
        """
        Check if a model is accessible and responding.

        Args:
            model: Model to check. Uses default if not provided.

        Returns:
            True if the model is healthy, False otherwise.
        """
        model = model or self.default_config.model
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5,
                timeout=10,
            )
            return response.choices[0].message.content is not None
        except Exception as e:
            logger.warning(f"Health check failed for {model}: {e}")
            return False

    async def cleanup(self) -> None:
        """
        Cleanup LiteLLM async clients.

        Call this method when shutting down to properly close async HTTP clients
        and prevent RuntimeWarnings about unawaited coroutines.

        Example:
            ```python
            client = LLMClient()
            # ... use client ...
            await client.cleanup()
            ```
        """
        try:
            # LiteLLM has a cleanup function for async clients
            if hasattr(litellm, "close_litellm_async_clients"):
                await litellm.close_litellm_async_clients()
                logger.debug("LiteLLM async clients closed")
        except Exception as e:
            logger.warning(f"Error cleaning up LiteLLM async clients: {e}")
