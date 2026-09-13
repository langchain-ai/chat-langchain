"""Helpers for extracting answer text from structured message content."""

from __future__ import annotations

from typing import Any


def content_part_text(part: Any) -> str:
    """Extract answer text from a content part."""
    if isinstance(part, dict):
        text = part.get("text")
        if isinstance(text, str) and (part.get("type") == "text" or "type" not in part):
            return text
        return ""
    if isinstance(part, list):
        return "\n".join(content_part_text(value) for value in part)
    return str(part)
