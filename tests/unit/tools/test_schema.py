"""Unit tests for tools schema normalization."""

import pytest

from orchestrator.tools.schema import (
    ensure_strict_json_schema,
    normalize_schema_for_llm,
)
import logging

logger = logging.getLogger(__name__)


class TestNormalizeSchema:
    def test_array_without_items(self):
        logger.info("NormalizeSchema: array without items")
        schema = {"type": "array"}
        result = normalize_schema_for_llm(schema)
        assert "items" in result

    def test_array_with_empty_items(self):
        logger.info("NormalizeSchema: array with empty items")
        schema = {"type": "array", "items": {}}
        result = normalize_schema_for_llm(schema)
        assert result["items"] != {}

    def test_array_with_valid_items(self):
        logger.info("NormalizeSchema: array with valid items")
        schema = {"type": "array", "items": {"type": "string"}}
        result = normalize_schema_for_llm(schema)
        assert result["items"]["type"] == "string"

    def test_object_without_properties(self):
        logger.info("NormalizeSchema: object without properties")
        schema = {"type": "object"}
        result = normalize_schema_for_llm(schema)
        assert "properties" in result

    def test_object_with_properties(self):
        logger.info("NormalizeSchema: object with properties")
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = normalize_schema_for_llm(schema)
        assert result["properties"]["name"]["type"] == "string"

    def test_nested_schema(self):
        logger.info("NormalizeSchema: nested schema")
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array"},
                "nested": {"type": "object"},
            },
        }
        result = normalize_schema_for_llm(schema)
        assert "items" in result["properties"]["items"]
        assert "properties" in result["properties"]["nested"]

    def test_infer_type_from_properties(self):
        logger.info("NormalizeSchema: infer type from properties")
        schema = {"properties": {"x": {"type": "string"}}}
        result = normalize_schema_for_llm(schema)
        assert result["type"] == "object"

    def test_infer_type_from_items(self):
        logger.info("NormalizeSchema: infer type from items")
        schema = {"items": {"type": "string"}}
        result = normalize_schema_for_llm(schema)
        assert result["type"] == "array"

    def test_anyof_normalization(self):
        logger.info("NormalizeSchema: anyof normalization")
        schema = {"anyOf": [{"type": "array"}, {"type": "object"}]}
        result = normalize_schema_for_llm(schema)
        assert "items" in result["anyOf"][0]

    def test_non_dict_passthrough(self):
        logger.info("NormalizeSchema: non dict passthrough")
        result = normalize_schema_for_llm("not a dict")
        assert result == "not a dict"


class TestEnsureStrictJsonSchema:
    def test_strict_mode_adds_required(self):
        logger.info("EnsureStrictJsonSchema: strict mode adds required")
        schema = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "int"}}}
        result = ensure_strict_json_schema(schema)
        assert "required" in result
        assert set(result["required"]) == {"a", "b"}
        assert result["additionalProperties"] is False

    def test_strict_mode_preserves_existing_required(self):
        logger.info("EnsureStrictJsonSchema: strict mode preserves existing required")
        schema = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
        result = ensure_strict_json_schema(schema)
        assert result["required"] == ["a"]

    def test_strict_mode_nested_objects(self):
        logger.info("EnsureStrictJsonSchema: strict mode nested objects")
        schema = {
            "type": "object",
            "properties": {
                "inner": {"type": "object", "properties": {"x": {"type": "int"}}},
            },
        }
        result = ensure_strict_json_schema(schema)
        assert result["additionalProperties"] is False
        assert "required" in result["properties"]["inner"]
