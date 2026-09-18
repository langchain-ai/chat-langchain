"""Detect and repair non-existent OpenAI model identifiers in generated code.

The docs agent grounds answers in retrieved doc pages, which sometimes contain
illustrative or placeholder model IDs (e.g. ``gpt-5.5``) that are not real,
copy-pasteable models. This helper flags those IDs so the answer path can
substitute a currently-supported model or note that the ID is illustrative.
"""

from __future__ import annotations

import re

#: Currently-supported model used when substituting an invalid identifier.
SUPPORTED_MODEL_SUBSTITUTE = "openai:gpt-4o"

#: Known non-existent / placeholder OpenAI model identifiers seen in doc snippets.
INVALID_MODEL_IDS = (
    "gpt-5.5-mini",
    "gpt-5.5-turbo",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-6-mini",
    "gpt-6-turbo",
    "gpt-6",
)

#: Matches any invalid ID, optionally prefixed by an ``openai:`` provider tag.
_INVALID_MODEL_RE = re.compile(
    r"(?:openai:)?(?:" + "|".join(re.escape(m) for m in INVALID_MODEL_IDS) + r")"
)


def find_invalid_model_ids(text: str) -> list[str]:
    """Return the invalid model identifiers referenced in ``text``."""
    if not text:
        return []
    return _INVALID_MODEL_RE.findall(text)


def contains_invalid_model_id(text: str) -> bool:
    """Return True when ``text`` references a known non-existent model ID."""
    return bool(_INVALID_MODEL_RE.search(text or ""))


def substitute_invalid_model_ids(text: str) -> str:
    """Replace known non-existent model IDs with a currently-supported model."""
    if not text:
        return text
    return _INVALID_MODEL_RE.sub(SUPPORTED_MODEL_SUBSTITUTE, text)


__all__ = [
    "SUPPORTED_MODEL_SUBSTITUTE",
    "INVALID_MODEL_IDS",
    "find_invalid_model_ids",
    "contains_invalid_model_id",
    "substitute_invalid_model_ids",
]
