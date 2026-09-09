"""Deterministic redaction for credentials in user-provided text."""

from __future__ import annotations

import math
import re
from collections import Counter

_PLACEHOLDER_PATTERN = re.compile(
    r"(?:YOUR_|xxx|<[^>]+>|\{\{[^}]+\}\}|example|placeholder|REDACTED|\*{3})",
    re.IGNORECASE,
)
_ASSIGNMENT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<key>api_key|token|secret|password|LANGSMITH_API_KEY|LANGCHAIN_API_KEY|X-Api-Key)"
    r"(?P<separator>\s*(?:=|:)\s*)(?P<quote>[\"']?)"
    r"(?P<value>[^\s,\"']+)(?P=quote)",
    re.IGNORECASE,
)
_SECRET_PATTERN = re.compile(
    r"(?P<token>"
    r"lsv2_(?:pt|sk)_[A-Za-z0-9._-]+|"
    r"lcl_[A-Za-z0-9._-]+|"
    r"sk-ant-[A-Za-z0-9_-]+|"
    r"sk-(?!ant-)[A-Za-z0-9_-]+|"
    r"AIza[A-Za-z0-9_-]+|"
    r"AKIA[A-Z0-9]+|"
    r"gh[opus]_[A-Za-z0-9]+|"
    r"xox(?:b|a|p|r|s)-[A-Za-z0-9-]+|"
    r"tvly-[A-Za-z0-9_-]+|"
    r"ey[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    r")"
)


def _is_placeholder(value: str) -> bool:
    """Return whether a value is an intentional placeholder."""
    return bool(_PLACEHOLDER_PATTERN.search(value))


def _has_high_entropy(value: str) -> bool:
    """Return whether an assignment value has enough entropy to redact."""
    if len(value) < 16 or _is_placeholder(value):
        return False
    classes = sum(
        bool(re.search(pattern, value))
        for pattern in (r"[a-z]", r"[A-Z]", r"\d", r"[^A-Za-z0-9]")
    )
    counts = Counter(value)
    entropy = -sum(
        (count / len(value)) * math.log2(count / len(value))
        for count in counts.values()
    )
    return classes >= 3 and entropy >= 3.0


def _placeholder(value: str) -> str:
    """Return a readable placeholder preserving a credential prefix."""
    if value.startswith("ey"):
        return "ey<REDACTED>"
    prefix_match = re.match(
        r"(?:lsv2_(?:pt|sk)_|lcl_|sk-ant-|sk-|AIza|AKIA|gh[opus]_|xox(?:b|a|p|r|s)-|tvly-)",
        value,
        re.IGNORECASE,
    )
    return f"{prefix_match.group(0) if prefix_match else ''}<REDACTED>"


def redact_secrets(text: str) -> tuple[str, int]:
    """Redact credential-shaped values and return the count of replacements."""
    matches: list[tuple[int, int, str, bool]] = []
    for match in _ASSIGNMENT_PATTERN.finditer(text):
        value = match.group("value")
        if _has_high_entropy(value):
            matches.append(
                (match.start("value"), match.end("value"), _placeholder(value), True)
            )
    for match in _SECRET_PATTERN.finditer(text):
        value = match.group("token")
        if not _is_placeholder(value):
            matches.append(
                (match.start("token"), match.end("token"), _placeholder(value), False)
            )

    replacements: list[tuple[int, int, str]] = []
    for start, end, replacement, contextual in sorted(
        matches, key=lambda item: (item[0], not item[3], -(item[1] - item[0]))
    ):
        if replacements and start < replacements[-1][1]:
            continue
        replacements.append((start, end, replacement))
    if not replacements:
        return text, 0
    pieces: list[str] = []
    cursor = 0
    for start, end, replacement in replacements:
        pieces.extend((text[cursor:start], replacement))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), len(replacements)


__all__ = ["redact_secrets"]
