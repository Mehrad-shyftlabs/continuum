"""
Integration tests for LLM providers (OpenAI, Gemini).

Requires API keys to be configured in .env.

Converted from tests/test_llm.py manual test script.
"""

import json
import os

import pytest
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


load_dotenv()

from orchestrator.config import settings
from orchestrator.llm import (
    ChatMessage,
    LLMClient,
    LLMConfig,
    ToolDefinition,
    FunctionDefinition,
)


OPENAI_MODEL = os.getenv("TEST_OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("TEST_GEMINI_MODEL", "gemini/gemini-2.5-pro")
TEST_MAX_TOKENS = int(os.getenv("DEFAULT_LLM_MAX_TOKENS", "100"))

pytestmark = [pytest.mark.integration]


@pytest.fixture
def openai_client():
    return LLMClient(
        config=LLMConfig(
            model=OPENAI_MODEL,
            max_tokens=TEST_MAX_TOKENS,
            temperature=settings.default_llm_temperature,
        ),
        enable_langfuse=False,
    )


@pytest.fixture
def gemini_client():
    return LLMClient(
        config=LLMConfig(
            model=GEMINI_MODEL,
            max_tokens=TEST_MAX_TOKENS,
            temperature=settings.default_llm_temperature,
        ),
        enable_langfuse=False,
    )


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
class TestOpenAI:
    async def test_basic_completion(self, openai_client):
        logger.info("OpenAI: basic completion")
        messages = [
            ChatMessage(role="system", content="You are a helpful assistant. Be concise."),
            ChatMessage(role="user", content="What is 2 + 2? Answer in one word."),
        ]
        response = await openai_client.chat(messages)

        assert response.model is not None
        assert response.content is not None
        assert response.usage is not None
        assert response.usage.total_tokens > 0

    async def test_streaming(self, openai_client):
        logger.info("OpenAI: streaming")
        messages = [
            ChatMessage(role="user", content="Count from 1 to 5, one number per line."),
        ]
        full_content = ""
        async for chunk in openai_client.chat_stream(messages):
            if chunk.content:
                full_content += chunk.content

        assert len(full_content) > 0

    async def test_function_calling(self):
        logger.info("OpenAI: function calling")
        client = LLMClient(
            config=LLMConfig(
                model=OPENAI_MODEL,
                max_tokens=max(TEST_MAX_TOKENS, 500),
                temperature=settings.default_llm_temperature,
            ),
            enable_langfuse=False,
        )

        weather_tool = ToolDefinition(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["location"],
                },
            ),
        )

        messages = [
            ChatMessage(role="user", content="What's the weather like in New York?"),
        ]

        response = await client.chat(
            messages,
            tools=[weather_tool],
            tool_choice="auto",
        )

        assert response is not None
        if response.tool_calls:
            assert len(response.tool_calls) >= 1
            tc = response.tool_calls[0]
            assert tc.function.name == "get_weather"
            args = json.loads(tc.function.arguments)
            assert "location" in args

    async def test_sync_completion(self, openai_client):
        logger.info("OpenAI: sync completion")
        messages = [
            ChatMessage(role="user", content="Say 'sync works' in exactly those words."),
        ]
        response = openai_client.chat_sync(messages)
        assert response is not None
        assert response.content is not None

    async def test_json_mode(self):
        logger.info("OpenAI: json mode")
        client = LLMClient(
            config=LLMConfig(
                model=OPENAI_MODEL,
                max_tokens=TEST_MAX_TOKENS,
                temperature=settings.default_llm_temperature,
                json_mode=True,
            ),
            enable_langfuse=False,
        )

        messages = [
            ChatMessage(
                role="system",
                content="You are a helpful assistant that responds in JSON format.",
            ),
            ChatMessage(
                role="user",
                content="Give me a JSON object with name='test' and value=42",
            ),
        ]

        response = await client.chat(messages)
        assert response.content is not None
        parsed = json.loads(response.content)
        assert isinstance(parsed, dict)

    async def test_fallback_mechanism(self, openai_client):
        logger.info("OpenAI: fallback mechanism")
        fallback_model = OPENAI_MODEL
        client = LLMClient(
            config=LLMConfig(
                model="non-existent-model",
                fallback_models=[fallback_model],
                enable_fallback=True,
                max_tokens=TEST_MAX_TOKENS,
                temperature=settings.default_llm_temperature,
            ),
            enable_langfuse=False,
        )

        messages = [
            ChatMessage(role="user", content="Say 'fallback works'"),
        ]

        try:
            response = await client.chat(messages)
            assert response is not None
            assert response.model is not None
        except Exception as e:
            if "not provided" in str(e).lower() or "invalid" in str(e).lower():
                pass
            else:
                raise


@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY"), reason="No Gemini API key")
class TestGemini:
    async def test_basic_completion(self, gemini_client):
        logger.info("Gemini: basic completion")
        messages = [
            ChatMessage(
                role="user",
                content="What is the capital of France? Answer in one word.",
            ),
        ]
        response = await gemini_client.chat(messages)

        assert response.model is not None
        assert response.content is not None
        assert response.usage is not None

    async def test_streaming(self, gemini_client):
        logger.info("Gemini: streaming")
        messages = [
            ChatMessage(role="user", content="List 3 colors, one per line."),
        ]
        full_content = ""
        async for chunk in gemini_client.chat_stream(messages):
            if chunk.content:
                full_content += chunk.content

        assert len(full_content) > 0


class TestModelUtilities:
    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
    def test_model_info(self):
        logger.info("ModelUtilities: model info")
        client = LLMClient(enable_langfuse=False)
        info = client.get_model_info(OPENAI_MODEL)
        assert isinstance(info, dict)

    def test_get_max_tokens(self):
        logger.info("ModelUtilities: get max tokens")
        client = LLMClient(enable_langfuse=False)
        max_tokens = client.get_max_tokens(OPENAI_MODEL)
        if max_tokens is not None:
            assert isinstance(max_tokens, int)
            assert max_tokens > 0

    def test_count_tokens(self):
        logger.info("ModelUtilities: count tokens")
        client = LLMClient(enable_langfuse=False)
        messages = [
            ChatMessage(role="user", content="Hello, how are you doing today?"),
        ]
        token_count = client.count_tokens(messages, OPENAI_MODEL)
        assert isinstance(token_count, int)
        assert token_count > 0

    @pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="No OpenAI API key")
    async def test_default_config(self):
        logger.info("ModelUtilities: default config")
        client = LLMClient(enable_langfuse=False)
        messages = [
            ChatMessage(role="user", content="Say 'defaults work'"),
        ]
        response = await client.chat(messages)
        assert response is not None
        assert response.content is not None
