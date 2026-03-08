"""
Unit tests for reasoning pattern features:
- Two-pass reasoning (reasoning_mode)
- ReAct mode (react_mode)
- ReflectionAgent
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orchestrator.agent.base import BaseAgent
from orchestrator.agent.config import AgentConfig, ReflectionConfig
from orchestrator.agent.types import AgentResponse, ResponseStatus, TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_agent(**kwargs) -> BaseAgent:
    return BaseAgent(name="test-agent", instructions="You are helpful.", **kwargs)


def make_llm_response(content: str = "ok", tokens: int = 10):
    resp = MagicMock()
    resp.content = content
    resp.tool_calls = None
    usage = MagicMock()
    usage.prompt_tokens = tokens
    usage.completion_tokens = tokens
    usage.total_tokens = tokens * 2
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# ReflectionConfig
# ---------------------------------------------------------------------------


class TestReflectionConfig:
    def test_defaults(self):
        cfg = ReflectionConfig()
        assert cfg.max_reflections == 2
        assert cfg.reflection_model is None
        assert "PASS" in cfg.critique_prompt

    def test_custom_values(self):
        cfg = ReflectionConfig(
            critique_prompt="Is this good? Reply PASS or NEEDS IMPROVEMENT.",
            max_reflections=5,
            reflection_model="gpt-4o",
        )
        assert cfg.max_reflections == 5
        assert cfg.reflection_model == "gpt-4o"


# ---------------------------------------------------------------------------
# AgentConfig — reasoning_mode and react_mode
# ---------------------------------------------------------------------------


class TestAgentConfigReasoningFields:
    def test_defaults_are_false(self):
        cfg = AgentConfig()
        assert cfg.reasoning_mode is False
        assert cfg.react_mode is False

    def test_can_enable_reasoning_mode(self):
        cfg = AgentConfig(reasoning_mode=True)
        assert cfg.reasoning_mode is True

    def test_can_enable_react_mode(self):
        cfg = AgentConfig(react_mode=True)
        assert cfg.react_mode is True

    def test_to_dict_includes_new_fields(self):
        cfg = AgentConfig(reasoning_mode=True, react_mode=True)
        d = cfg.to_dict()
        assert d["reasoning_mode"] is True
        assert d["react_mode"] is True

    def test_to_dict_defaults_are_false(self):
        d = AgentConfig().to_dict()
        assert d["reasoning_mode"] is False
        assert d["react_mode"] is False


# ---------------------------------------------------------------------------
# MessageBuilder — react_mode injects template
# ---------------------------------------------------------------------------


class TestMessageBuilderReactMode:
    @pytest.mark.asyncio
    async def test_react_mode_injects_template(self):
        from orchestrator.agent.execution.message_builder import MessageBuilder, _REACT_TEMPLATE

        builder = MessageBuilder()
        agent = make_agent(config=AgentConfig(react_mode=True))

        # Minimal RunContext mock
        ctx = MagicMock()
        ctx.session_id = None

        # Patch memory + session services (not set, so no-ops)
        messages = await builder.prepare_messages(
            agent=agent,
            input="hello",
            context=ctx,
        )

        system_contents = [m["content"] for m in messages if m.get("role") == "system"]
        assert any("Thought:" in c for c in system_contents), (
            "ReAct template should be injected as a system message"
        )

    @pytest.mark.asyncio
    async def test_react_mode_off_does_not_inject(self):
        from orchestrator.agent.execution.message_builder import MessageBuilder

        builder = MessageBuilder()
        agent = make_agent(config=AgentConfig(react_mode=False))

        ctx = MagicMock()
        ctx.session_id = None

        messages = await builder.prepare_messages(
            agent=agent,
            input="hello",
            context=ctx,
        )

        system_contents = [m["content"] for m in messages if m.get("role") == "system"]
        assert not any("Thought:" in c for c in system_contents)


# ---------------------------------------------------------------------------
# Executor — reasoning_mode calls _run_reasoning_pass
# ---------------------------------------------------------------------------


class TestExecutorReasoningMode:
    @pytest.mark.asyncio
    async def test_reasoning_mode_injects_reasoning_block(self):
        from orchestrator.agent.execution.executor import Executor

        # Build a minimal executor with a mocked LLM client
        llm_client = AsyncMock()
        # First call = reasoning pass, second call = main turn
        llm_client.chat.side_effect = [
            make_llm_response("step-by-step thoughts"),  # reasoning pass
            make_llm_response("final answer"),           # main turn
        ]

        executor = Executor(llm_client=llm_client)

        agent = make_agent(config=AgentConfig(reasoning_mode=True))
        # Patch LLMConfig.from_agent_config to return a simple mock
        with patch("orchestrator.agent.execution.executor.LLMConfig") as mock_cfg_cls:
            mock_cfg = MagicMock()
            mock_cfg.max_tokens = None
            mock_cfg_cls.from_agent_config.return_value = mock_cfg

            messages = [{"role": "user", "content": "What is 2+2?"}]
            run_state = MagicMock()
            run_state.messages = messages
            run_state.turn_count = 0

            ctx = MagicMock()
            ctx.session_id = "sid"
            ctx.max_turns = 10
            ctx.metadata = {}

            response = await executor.execute_loop(
                agent=agent,
                messages=messages,
                context=ctx,
                run_state=run_state,
            )

        # LLM should have been called twice
        assert llm_client.chat.call_count == 2

        # The reasoning block should have been injected into the messages passed to the 2nd call
        second_call_messages = llm_client.chat.call_args_list[1][1]["messages"]
        reasoning_msgs = [
            m for m in second_call_messages
            if m.get("role") == "system" and "<reasoning>" in m.get("content", "")
        ]
        assert reasoning_msgs, "A <reasoning> system message should be injected"
        assert "step-by-step thoughts" in reasoning_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_reasoning_mode_off_no_extra_call(self):
        from orchestrator.agent.execution.executor import Executor

        llm_client = AsyncMock()
        llm_client.chat.return_value = make_llm_response("answer")

        executor = Executor(llm_client=llm_client)
        agent = make_agent(config=AgentConfig(reasoning_mode=False))

        with patch("orchestrator.agent.execution.executor.LLMConfig") as mock_cfg_cls:
            mock_cfg = MagicMock()
            mock_cfg_cls.from_agent_config.return_value = mock_cfg

            messages = [{"role": "user", "content": "Hi"}]
            run_state = MagicMock()
            run_state.messages = messages
            run_state.turn_count = 0

            ctx = MagicMock()
            ctx.session_id = None
            ctx.max_turns = 10
            ctx.metadata = {}

            await executor.execute_loop(
                agent=agent,
                messages=messages,
                context=ctx,
                run_state=run_state,
            )

        # Only the single main-turn call
        assert llm_client.chat.call_count == 1


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------


class TestReflectionAgent:
    def test_import(self):
        from orchestrator.agent.workflow.reflection import ReflectionAgent

        assert ReflectionAgent is not None
        assert issubclass(ReflectionAgent, BaseAgent)

    def test_create_reflection_agent_factory(self):
        from orchestrator.agent.workflow.reflection import create_reflection_agent

        inner = make_agent()
        agent = create_reflection_agent(
            name="my-reflector",
            agent=inner,
            max_reflections=3,
            reflection_model="gpt-4o-mini",
        )
        assert agent.name == "my-reflector"
        assert agent.reflection_config.max_reflections == 3
        assert agent.reflection_config.reflection_model == "gpt-4o-mini"

    def test_requires_inner_agent(self):
        from orchestrator.agent.exceptions import AgentConfigurationError
        from orchestrator.agent.workflow.reflection import ReflectionAgent

        with pytest.raises(AgentConfigurationError):
            ReflectionAgent(name="r")

    @pytest.mark.asyncio
    async def test_passes_on_first_try(self):
        """When critique returns PASS immediately, inner agent runs once."""
        from orchestrator.agent.workflow.reflection import ReflectionAgent

        inner = make_agent()
        reflector = ReflectionAgent(name="reflector", agent=inner)

        runner = AsyncMock()
        runner.run.return_value = AgentResponse(
            content="great answer",
            agent_name="test-agent",
            status=ResponseStatus.SUCCESS,
            usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

        llm_client = AsyncMock()
        llm_client.chat.return_value = make_llm_response("PASS")

        ctx = MagicMock()
        result = await reflector.execute(
            input_text="What is AI?",
            runner=runner,
            context=ctx,
            llm_client=llm_client,
        )

        assert result.content == "great answer"
        assert runner.run.call_count == 1
        assert llm_client.chat.call_count == 1  # one critique call

    @pytest.mark.asyncio
    async def test_retries_on_needs_improvement(self):
        """When critique returns NEEDS IMPROVEMENT, inner agent is retried."""
        from orchestrator.agent.workflow.reflection import ReflectionAgent

        inner = make_agent()
        reflector = ReflectionAgent(
            name="reflector",
            agent=inner,
            reflection_config=ReflectionConfig(max_reflections=2),
        )

        runner = AsyncMock()
        runner.run.side_effect = [
            AgentResponse(
                content="first attempt",
                agent_name="test-agent",
                status=ResponseStatus.SUCCESS,
                usage=TokenUsage(5, 5, 10),
            ),
            AgentResponse(
                content="second attempt",
                agent_name="test-agent",
                status=ResponseStatus.SUCCESS,
                usage=TokenUsage(5, 5, 10),
            ),
        ]

        llm_client = AsyncMock()
        # First critique: needs improvement; second run should trigger second critique → pass
        llm_client.chat.side_effect = [
            make_llm_response("NEEDS IMPROVEMENT: too brief"),
            make_llm_response("PASS"),
        ]

        ctx = MagicMock()
        result = await reflector.execute(
            input_text="Explain quantum computing",
            runner=runner,
            context=ctx,
            llm_client=llm_client,
        )

        assert result.content == "second attempt"
        assert runner.run.call_count == 2

    @pytest.mark.asyncio
    async def test_stops_after_max_reflections(self):
        """Never runs more than max_reflections + 1 times even if critique never passes."""
        from orchestrator.agent.workflow.reflection import ReflectionAgent

        inner = make_agent()
        reflector = ReflectionAgent(
            name="reflector",
            agent=inner,
            reflection_config=ReflectionConfig(max_reflections=2),
        )

        runner = AsyncMock()
        runner.run.return_value = AgentResponse(
            content="attempt",
            agent_name="test-agent",
            status=ResponseStatus.SUCCESS,
            usage=TokenUsage(5, 5, 10),
        )

        llm_client = AsyncMock()
        llm_client.chat.return_value = make_llm_response("NEEDS IMPROVEMENT: still bad")

        ctx = MagicMock()
        result = await reflector.execute(
            input_text="hard question",
            runner=runner,
            context=ctx,
            llm_client=llm_client,
        )

        # max_reflections=2 → at most 3 runs (initial + 2 retries)
        assert runner.run.call_count == 3
        assert result.content == "attempt"

    def test_exported_from_workflow_init(self):
        from orchestrator.agent.workflow import ReflectionAgent, create_reflection_agent

        assert ReflectionAgent is not None
        assert callable(create_reflection_agent)

    def test_exported_from_agent_init(self):
        from orchestrator.agent import ReflectionAgent, ReflectionConfig, create_reflection_agent

        assert ReflectionAgent is not None
        assert ReflectionConfig is not None
        assert callable(create_reflection_agent)
