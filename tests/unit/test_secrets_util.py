"""Unit tests for secrets utility functions."""

import pytest

from orchestrator.utils.secrets import (
    is_sensitive_key,
    mask_value,
    redact_dict,
    redact_sensitive_values,
)
import logging

logger = logging.getLogger(__name__)


class TestMaskValue:
    def test_mask_value_normal(self):
        logger.info("MaskValue: mask value normal")
        assert mask_value("sk-1234567890abcdef") == "****cdef"

    def test_mask_value_short_string(self):
        logger.info("MaskValue: mask value short string")
        assert mask_value("abc") == "****"

    def test_mask_value_empty(self):
        logger.info("MaskValue: mask value empty")
        assert mask_value("") == "****"

    def test_mask_value_custom_visible_chars(self):
        logger.info("MaskValue: mask value custom visible chars")
        assert mask_value("sk-1234567890abcdef", visible_chars=6) == "****abcdef"

    def test_mask_value_exact_length(self):
        logger.info("MaskValue: mask value exact length")
        assert mask_value("abcd", visible_chars=4) == "****"

    def test_mask_value_longer_than_visible(self):
        logger.info("MaskValue: mask value longer than visible")
        assert mask_value("abcde", visible_chars=4) == "****bcde"


class TestIsSensitiveKey:
    def test_api_key(self):
        logger.info("IsSensitiveKey: api key")
        assert is_sensitive_key("api_key") is True
        assert is_sensitive_key("openai_api_key") is True
        assert is_sensitive_key("OPENAI_API_KEY") is True

    def test_password(self):
        logger.info("IsSensitiveKey: password")
        assert is_sensitive_key("password") is True
        assert is_sensitive_key("session_redis_password") is True

    def test_token(self):
        logger.info("IsSensitiveKey: token")
        assert is_sensitive_key("auth_token") is True
        assert is_sensitive_key("access_token") is True
        assert is_sensitive_key("bearer") is True

    def test_secret(self):
        logger.info("IsSensitiveKey: secret")
        assert is_sensitive_key("secret") is True
        assert is_sensitive_key("langfuse_secret_key") is True

    def test_normal_field(self):
        logger.info("IsSensitiveKey: normal field")
        assert is_sensitive_key("model") is False
        assert is_sensitive_key("temperature") is False
        assert is_sensitive_key("host") is False
        assert is_sensitive_key("port") is False


class TestRedactDict:
    def test_redact_dict_basic(self):
        logger.info("RedactDict: redact dict basic")
        data = {"api_key": "sk-12345678901234567890", "model": "gpt-4"}
        result = redact_dict(data)
        assert result["api_key"] == "****7890"
        assert result["model"] == "gpt-4"

    def test_redact_dict_nested(self):
        logger.info("RedactDict: redact dict nested")
        data = {
            "config": {
                "api_key": "sk-secret-value-here",
                "name": "test",
            }
        }
        result = redact_dict(data)
        assert result["config"]["api_key"] == "****here"
        assert result["config"]["name"] == "test"

    def test_redact_dict_non_string_sensitive(self):
        logger.info("RedactDict: redact dict non string sensitive")
        data = {"token": 12345}
        result = redact_dict(data)
        assert result["token"] == "[REDACTED]"

    def test_redact_dict_max_depth(self):
        logger.info("RedactDict: redact dict max depth")
        data = {"level1": {"level2": {"level3": {"api_key": "secret"}}}}
        result = redact_dict(data, max_depth=2)
        assert result["level1"]["level2"]["level3"]["api_key"] == "secret"

    def test_redact_dict_with_list(self):
        logger.info("RedactDict: redact dict with list")
        data = {"items": [{"api_key": "sk-secretvalue1234"}, {"name": "ok"}]}
        result = redact_dict(data)
        assert result["items"][0]["api_key"] == "****1234"
        assert result["items"][1]["name"] == "ok"


class TestRedactSensitiveValues:
    def test_redact_openai_key(self):
        logger.info("RedactSensitiveValues: redact openai key")
        text = "Error with key sk-1234567890abcdefghijklmn"
        result = redact_sensitive_values(text)
        assert "sk-1234567890" not in result
        assert "[REDACTED]" in result

    def test_redact_bearer_token(self):
        logger.info("RedactSensitiveValues: redact bearer token")
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
        result = redact_sensitive_values(text)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result
        assert "[REDACTED]" in result

    def test_redact_langfuse_keys(self):
        logger.info("RedactSensitiveValues: redact langfuse keys")
        text = "pk-lf-abc123def456 and sk-lf-xyz789ghi012"
        result = redact_sensitive_values(text)
        assert "pk-lf-abc123def456" not in result
        assert "sk-lf-xyz789ghi012" not in result

    def test_no_redaction_normal_text(self):
        logger.info("RedactSensitiveValues: no redaction normal text")
        text = "This is a normal message with no secrets"
        assert redact_sensitive_values(text) == text


class TestSettingsRepr:
    def test_settings_repr_masks_keys(self):
        logger.info("SettingsRepr: settings repr masks keys")
        from orchestrator.config import Settings

        s = Settings(
            openai_api_key="sk-1234567890abcdefghij",
            langfuse_secret_key="sk-lf-test-secret-value",
        )
        repr_str = repr(s)
        assert "sk-1234567890" not in repr_str
        assert "sk-lf-test-secret" not in repr_str
        assert "****" in repr_str


class TestErrorContextRedacted:
    def test_error_context_redacted_before_langfuse(self):
        logger.info("ErrorContextRedacted: error context redacted before langfuse")
        from orchestrator.exceptions import OrchestratorError

        error = OrchestratorError(
            "test error",
            context={"api_key": "sk-secretvaluehere1234", "model": "gpt-4"},
            should_report=False,
        )
        d = error.to_dict()
        assert "sk-secretvaluehere" not in str(d)
        assert d["context"]["model"] == "gpt-4"

    def test_error_str_redacts_context(self):
        logger.info("ErrorContextRedacted: error str redacts context")
        from orchestrator.exceptions import OrchestratorError

        error = OrchestratorError(
            "Failed with key sk-1234567890abcdefghijklm",
            context={"api_key": "sk-1234567890abcdefghijklm"},
            should_report=False,
        )
        s = str(error)
        assert "sk-1234567890" not in s
        assert "[REDACTED]" in s


class TestToolContextStateMasksSensitive:
    def test_tool_context_state_masks_sensitive(self):
        logger.info("ToolContextStateMasksSensitive: tool context state masks sensitive")
        from orchestrator.tools.types import ToolContextState

        state = ToolContextState()
        state.set("ns1", "session_id", "sess-123")
        state.set("ns1", "auth_token", "super-secret-token-value", sensitive=True)

        d = state.to_dict()
        assert d["variables"]["ns1"]["session_id"] == "sess-123"
        assert "super-secret-token" not in d["variables"]["ns1"]["auth_token"]
        assert "****" in d["variables"]["ns1"]["auth_token"]

    def test_to_prompt_context_excludes_sensitive(self):
        logger.info("ToolContextStateMasksSensitive: to prompt context excludes sensitive")
        from orchestrator.tools.types import ToolContextState

        state = ToolContextState()
        state.set("ns1", "session_id", "sess-123")
        state.set("ns1", "auth_token", "super-secret-token-value", sensitive=True)

        prompt = state.to_prompt_context()
        assert "session_id" in prompt
        assert "auth_token" not in prompt
        assert "super-secret" not in prompt
