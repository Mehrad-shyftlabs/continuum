"""
Executor - Core execution logic for agents.

Extracted from AgentRunner to provide clean separation of concerns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from orchestrator.agent.exceptions import MaxTurnsExceededError
from orchestrator.agent.interfaces.executor_interface import IExecutor
from orchestrator.agent.types import (
    AgentResponse,
    ResponseStatus,
    RunContext,
    RunState,
    TokenUsage,
    ToolExecutionSummary,
)
from orchestrator.llm.config import LLMConfig
from orchestrator.logging import get_logger
from orchestrator.observability.metrics import get_metrics_collector
from orchestrator.observability.trace_context import SpanScope, truncate_data

if TYPE_CHECKING:
    from orchestrator.agent.base import BaseAgent
    from orchestrator.agent.execution.handoff_executor import HandoffExecutor
    from orchestrator.agent.execution.tool_handler import ToolHandler
    from orchestrator.llm import LLMClient

logger = get_logger(__name__)


class Executor(IExecutor):
    """
    Core executor for agent runs.

    Handles the main conversation loop with tool calls and handoffs.
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        tool_handler: ToolHandler | None = None,
        handoff_executor: HandoffExecutor | None = None,
    ):
        """
        Initialize executor.

        Args:
            llm_client: LLM client for model calls
            tool_handler: Tool handler for tool execution
            handoff_executor: Handoff executor for agent handoffs
        """
        self._llm_client = llm_client
        self._tool_handler = tool_handler
        self._handoff_executor = handoff_executor

        # Set executor reference in handoff executor for recursive execution
        if self._handoff_executor and hasattr(self._handoff_executor, "_executor"):
            self._handoff_executor._executor = self

    @property
    def llm_client(self) -> LLMClient:
        """Get LLM client."""
        if not self._llm_client:
            raise RuntimeError("LLMClient not provided to Executor")
        return self._llm_client

    async def execute_loop(
        self,
        agent: BaseAgent,
        messages: list[dict[str, Any]],
        context: RunContext,
        run_state: RunState,
    ) -> AgentResponse:
        """
        Execute the main conversation loop.

        Args:
            agent: Agent to execute
            messages: Initial messages
            context: Run context
            run_state: Run state

        Returns:
            AgentResponse with the result
        """
        turn = 0
        total_usage = TokenUsage()
        metrics = get_metrics_collector()

        # Collect tool execution summaries for session storage
        all_tool_summaries: list[ToolExecutionSummary] = []

        while turn < context.max_turns:
            turn += 1
            run_state.turn_count = turn

            # Create span for this turn
            async with SpanScope(
                f"turn.{turn}",
                input=truncate_data(
                    {
                        "turn": turn,
                        "message_count": len(messages),
                        "last_message_role": messages[-1].get("role") if messages else None,
                    }
                ),
                metadata={
                    "agent_name": agent.name,
                    "turn_number": turn,
                    "max_turns": context.max_turns,
                },
            ) as turn_span:
                # Get tools including handoffs
                tools = agent.get_tools_for_llm()
                tool_names = [t.get("function", {}).get("name", "") for t in tools] if tools else []
                turn_span.add_metadata("available_tools", tool_names[:20])

                # Make LLM call
                try:
                    # Create LLMConfig for this agent (includes JSON mode if enabled)
                    llm_config = LLMConfig.from_agent_config(agent)

                    # Log JSON mode status
                    if agent.enable_json_mode:
                        json_mode_info = "enabled"
                        if agent.json_schema:
                            if isinstance(agent.json_schema, type):
                                json_mode_info += f" with schema: {agent.json_schema.__name__}"
                            else:
                                json_mode_info += " with JSON schema dict"
                        else:
                            json_mode_info += " (simple json_object mode)"
                        logger.info(
                            f"📋 JSON mode {json_mode_info} for agent {agent.name}",
                            extra={
                                "agent_name": agent.name,
                                "json_mode": True,
                                "json_schema": (
                                    agent.json_schema.__name__
                                    if isinstance(agent.json_schema, type)
                                    else "dict"
                                    if isinstance(agent.json_schema, dict)
                                    else None
                                ),
                            },
                        )

                    # NOTE: We pass auto_session=False because Executor manages the
                    # conversation loop including tool calls.
                    response = await self.llm_client.chat(
                        messages=messages,
                        tools=tools if tools else None,
                        config=llm_config,
                        session_id=context.session_id,
                        trace_metadata={"session_id": context.session_id}
                        if context.session_id
                        else None,
                        auto_session=False,  # Executor manages the message loop
                    )
                except Exception as e:
                    turn_span.set_error(str(e))
                    raise

                # Track usage
                if response.usage:
                    total_usage = total_usage.add(
                        TokenUsage(
                            prompt_tokens=response.usage.prompt_tokens or 0,
                            completion_tokens=response.usage.completion_tokens or 0,
                            total_tokens=response.usage.total_tokens or 0,
                        )
                    )
                    turn_span.add_metadata(
                        "tokens",
                        {
                            "prompt": response.usage.prompt_tokens,
                            "completion": response.usage.completion_tokens,
                            "total": response.usage.total_tokens,
                        },
                    )

                    # Track token usage in metrics
                    metrics.track_tokens(
                        f"turn_{turn}_llm",
                        prompt_tokens=response.usage.prompt_tokens or 0,
                        completion_tokens=response.usage.completion_tokens or 0,
                        model=agent.model,
                    )

                # Add assistant message
                assistant_msg = {
                    "role": "assistant",
                    "content": response.content,
                }
                if response.tool_calls:
                    assistant_msg["tool_calls"] = [
                        tc.to_dict() if hasattr(tc, "to_dict") else tc for tc in response.tool_calls
                    ]
                messages.append(assistant_msg)
                from orchestrator.agent.utils.message_utils import message_to_dict

                run_state.messages = [message_to_dict(m) for m in messages]

                # Log LLM response details
                if not response.tool_calls:
                    logger.debug(
                        f"💬 LLM response (no tool calls) on turn {turn}: "
                        f"content_preview={(response.content or '')[:150]}, "
                        f"messages_in_context={len(messages)}"
                    )
                else:
                    tool_names = [
                        tc.function.name
                        if hasattr(tc, "function")
                        else tc.get("function", {}).get("name", "")
                        for tc in response.tool_calls
                    ]
                    logger.info(
                        f"🔧 LLM response (with {len(response.tool_calls)} tool calls) on turn {turn}: {', '.join(tool_names)}"
                    )

                # Handle tool calls
                if response.tool_calls:
                    called_tool_names = [
                        tc.function.name
                        if hasattr(tc, "function")
                        else tc.get("function", {}).get("name", "")
                        for tc in response.tool_calls
                    ]
                    turn_span.add_metadata("tool_calls", called_tool_names)
                    logger.info(
                        f"🤖 LLM requesting {len(response.tool_calls)} tool(s): {', '.join(called_tool_names)}"
                    )

                    # Separate handoffs from regular tools
                    handoff_calls = []
                    regular_tool_calls = []

                    for tc in response.tool_calls:
                        tool_name = (
                            tc.function.name
                            if hasattr(tc, "function")
                            else tc.get("function", {}).get("name", "")
                        )
                        is_handoff, target = agent.is_handoff_tool_call(tool_name)
                        if is_handoff and target:
                            handoff_calls.append((tc, target))
                        else:
                            regular_tool_calls.append(tc)

                    # Execute regular tools
                    if regular_tool_calls and self._tool_handler:
                        # Create summary for this turn's tool executions
                        turn_tool_summary = ToolExecutionSummary()

                        tool_results = await self._tool_handler.execute_tools_batch(
                            agent=agent,
                            tool_calls=regular_tool_calls,
                            context=context,
                            tool_summary=turn_tool_summary,
                        )
                        messages.extend(tool_results)

                        # Store the summary if tools were executed
                        if not turn_tool_summary.is_empty():
                            all_tool_summaries.append(turn_tool_summary)

                    # Execute handoffs sequentially (they may return early)
                    if handoff_calls and self._handoff_executor:
                        for tc, target in handoff_calls:
                            handoff_result = await self._handoff_executor.execute_handoff(
                                agent=agent,
                                target_name=target,
                                tool_call=tc,
                                messages=messages,
                                context=context,
                                run_state=run_state,
                            )

                            if handoff_result.success and handoff_result.response:
                                # Handoff was executed and has response
                                turn_span.set_output({"handoff_to": target, "success": True})
                                return AgentResponse(
                                    content=handoff_result.response.content,
                                    agent_name=handoff_result.response.agent_name,
                                    status=ResponseStatus.SUCCESS,
                                    usage=total_usage.add(handoff_result.response.usage),
                                    turn_count=turn,
                                    handoff_result=handoff_result,
                                    messages=messages,
                                )
                            else:
                                # Handoff failed, add error as tool result
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": (
                                            tc.id if hasattr(tc, "id") else tc.get("id", "")
                                        ),
                                        "content": f"Handoff failed: {handoff_result.error or 'Unknown error'}",
                                    }
                                )

                    # Update state
                    from orchestrator.agent.utils.message_utils import message_to_dict

                    run_state.messages = [message_to_dict(m) for m in messages]

                    # Update span with tool execution summary
                    turn_span.set_output(
                        {
                            "tool_calls_executed": len(regular_tool_calls),
                            "handoffs_attempted": len(handoff_calls),
                            "continuing_to_next_turn": True,
                        }
                    )

                    # Continue to next turn
                    continue

                # No tool calls, we're done
                turn_span.set_output(
                    {
                        "response_preview": (response.content or "")[:200],
                        "final_turn": True,
                    }
                )

                # Merge all tool summaries into one for the response
                merged_tool_summary = self._merge_tool_summaries(all_tool_summaries)

                # Parse structured output if JSON mode was enabled and output_schema is set
                structured_output = None
                if agent.enable_json_mode and agent.output_schema and response.content:
                    try:
                        import json

                        # Log that we're expecting JSON format
                        logger.info(
                            f"🔍 Verifying JSON format response for agent {agent.name}",
                            extra={
                                "agent_name": agent.name,
                                "content_length": len(response.content) if response.content else 0,
                                "output_schema": agent.output_schema.__name__,
                            },
                        )

                        # Check if content looks like JSON
                        content_stripped = response.content.strip()
                        is_json_like = (
                            content_stripped.startswith("{") and content_stripped.endswith("}")
                        ) or (content_stripped.startswith("[") and content_stripped.endswith("]"))

                        if not is_json_like:
                            logger.warning(
                                f"⚠️ Response from agent {agent.name} doesn't appear to be JSON format "
                                f"(expected JSON mode enabled). Content preview: {content_stripped[:100]}",
                                extra={
                                    "agent_name": agent.name,
                                    "content_preview": content_stripped[:200],
                                },
                            )

                        # Parse JSON content
                        parsed_json = json.loads(response.content)
                        logger.info(
                            f"✅ Successfully parsed JSON response for agent {agent.name}",
                            extra={
                                "agent_name": agent.name,
                                "json_keys": list(parsed_json.keys()) if isinstance(parsed_json, dict) else None,
                            },
                        )

                        # Validate against output_schema Pydantic model
                        structured_output = agent.output_schema.model_validate(parsed_json)
                        logger.info(
                            f"✅ Successfully validated structured output for agent {agent.name} against {agent.output_schema.__name__}",
                            extra={
                                "agent_name": agent.name,
                                "output_schema": agent.output_schema.__name__,
                            },
                        )
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"❌ Failed to parse JSON response for agent {agent.name}: {e}",
                            extra={
                                "agent_name": agent.name,
                                "content_preview": response.content[:200] if response.content else None,
                                "error": str(e),
                            },
                        )
                    except Exception as e:
                        logger.error(
                            f"❌ Failed to validate structured output against schema for agent {agent.name}: {e}",
                            extra={
                                "agent_name": agent.name,
                                "output_schema": agent.output_schema.__name__ if agent.output_schema else None,
                                "error": str(e),
                            },
                        )
                elif agent.enable_json_mode and response.content:
                    # JSON mode enabled but no output_schema - just verify it's JSON
                    import json

                    try:
                        content_stripped = response.content.strip()
                        parsed_json = json.loads(response.content)
                        logger.info(
                            f"✅ JSON mode enabled: Response is valid JSON for agent {agent.name}",
                            extra={
                                "agent_name": agent.name,
                                "json_keys": list(parsed_json.keys()) if isinstance(parsed_json, dict) else None,
                            },
                        )
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"⚠️ JSON mode enabled but response is not valid JSON for agent {agent.name}: {e}",
                            extra={
                                "agent_name": agent.name,
                                "content_preview": content_stripped[:200] if response.content else None,
                            },
                        )

                # No tool calls, we're done
                agent_response = AgentResponse(
                    content=response.content or "",
                    structured_output=structured_output,
                    agent_name=agent.name,
                    status=ResponseStatus.SUCCESS,
                    usage=total_usage,
                    turn_count=turn,
                    messages=messages,
                )
                # Store tool summary in metadata for session storage
                if merged_tool_summary and not merged_tool_summary.is_empty():
                    context.metadata["tool_execution_summary"] = merged_tool_summary.to_dict()

                return agent_response

        # Max turns exceeded
        raise MaxTurnsExceededError(
            max_turns=context.max_turns,
            current_turn=turn,
            agent_name=agent.name,
            run_id=context.run_id,
        )

    def _merge_tool_summaries(
        self,
        summaries: list[ToolExecutionSummary],
    ) -> ToolExecutionSummary | None:
        """Merge multiple turn tool summaries into one."""
        if not summaries:
            return None

        merged = ToolExecutionSummary()

        for summary in summaries:
            merged.tools_used.extend(summary.tools_used)
            merged.tool_count += summary.tool_count
            merged.total_latency_ms += summary.total_latency_ms
            merged.tool_latencies.update(summary.tool_latencies)
            merged.success_count += summary.success_count
            merged.error_count += summary.error_count
            merged.errors.extend(summary.errors)
            merged.input_tokens += summary.input_tokens
            merged.output_tokens += summary.output_tokens

            # Merge servers (unique)
            for server in summary.servers_used:
                if server not in merged.servers_used:
                    merged.servers_used.append(server)

            # Merge auth info
            merged.auth_info.update(summary.auth_info)

        return merged
