"""Utility functions for masking and redacting sensitive data."""

from __future__ import annotations

import re
from typing import Any

SENSITIVE_KEY_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|password|auth|credential|bearer)"),
    re.compile(r"(?i)(access[_-]?key|private[_-]?key|session[_-]?secret)"),
]

SENSITIVE_VALUE_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),
    re.compile(r"pk-lf-[a-zA-Z0-9]+"),
    re.compile(r"sk-lf-[a-zA-Z0-9]+"),
    re.compile(r"Bearer\s+[a-zA-Z0-9._-]+"),
]


def mask_value(value: str, visible_chars: int = 4) -> str:
    """Mask a sensitive string, showing only last N characters."""
    if not value or len(value) <= visible_chars:
        return "****"
    return f"****{value[-visible_chars:]}"


def is_sensitive_key(key: str) -> bool:
    """Check if a dictionary key likely contains sensitive data."""
    return any(pattern.search(key) for pattern in SENSITIVE_KEY_PATTERNS)


def redact_dict(data: dict[str, Any], depth: int = 0, max_depth: int = 5) -> dict[str, Any]:
    """Recursively redact sensitive values from a dictionary."""
    if depth > max_depth:
        return data

    result = {}
    for key, value in data.items():
        if is_sensitive_key(key):
            if isinstance(value, str):
                result[key] = mask_value(value)
            else:
                result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = redact_dict(value, depth + 1, max_depth)
        elif isinstance(value, list):
            result[key] = [
                redact_dict(item, depth + 1, max_depth) if isinstance(item, dict) else item
                for item in value
            ]
        elif isinstance(value, str):
            result[key] = redact_sensitive_values(value)
        else:
            result[key] = value
    return result


def redact_sensitive_values(text: str) -> str:
    """Redact known sensitive patterns from a text string."""
    result = text
    for pattern in SENSITIVE_VALUE_PATTERNS:
        result = pattern.sub("[REDACTED]", result)
    return result
