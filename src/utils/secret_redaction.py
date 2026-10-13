"""Deterministic credential redaction for messages and LangSmith trace writes."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from langsmith.anonymizer import create_anonymizer

REDACTED_SECRET = "[REDACTED_SECRET]"

_VALUE = r"""(?:"[^"\n]*"|'[^'\n]*'|<[^>\n]*>|[^\s"'`,;]+)"""
_SECRET_PATTERN = re.compile(
    rf"(?P<assignment>(?i:\baws_secret_access_key\b[\"']?\s*[:=]\s*|\b[A-Z_][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD)\s*=\s*|\b(?:KEY|TOKEN|SECRET|PASSWORD)\s*=\s*))(?P<assigned>{_VALUE})"
    rf"|(?P<bearer>(?i:\bBearer\s+))(?P<authorization>{_VALUE})"
    r"|(?P<postgres>(?i:\bpostgres(?:ql)?://)[^\s/@:]*:)(?P<password><[^>\n]*>|[^\s@/?#]+)(?=@)"
    r"|(?<![A-Za-z0-9])(?P<token>"
    r"lsv2_(?:pt|sk)_[A-Za-z0-9_]+"
    r"|lcl_[A-Za-z0-9_]+"
    r"|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"
    r"|AIza[0-9A-Za-z_-]{30,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}"
    r"|xox[baprs]-[A-Za-z0-9-]{10,}"
    r"|tvly-[A-Za-z0-9_-]{10,}"
    r"|(?:sk|rk)_live_[A-Za-z0-9]+"
    r"|AKIA[0-9A-Z]{16}"
    r"|eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)"
)
_TOKEN_PREFIX = re.compile(
    r"^(?:lsv2_(?:pt|sk)_|lcl_|sk-(?:proj-|ant-)?|xox[baprs]-|tvly-|(?:sk|rk)_live_)"
)


def _is_placeholder(value: str) -> bool:
    """Recognize documented placeholders and unevaluated variable references."""
    value = value.strip("\"'")
    candidate = _TOKEN_PREFIX.sub("", value).lower()
    return candidate.startswith(
        ("your_", "your-", "[redacted_secret]")
    ) or value.startswith(("<", "$", "settings.", "os.environ[", "os.getenv("))


def redact_secrets(text: str) -> tuple[str, int]:
    """Replace credential values while retaining assignment names and placeholders."""
    replacements = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal replacements
        value = (
            match.group("assigned")
            or match.group("authorization")
            or match.group("password")
            or match.group("token")
        )
        if _is_placeholder(value):
            return match.group(0)
        replacements += 1
        prefix = (
            match.group("assignment")
            or match.group("bearer")
            or match.group("postgres")
            or ""
        )
        quote = value[0] if not match.group("postgres") and value[0] in "\"'" else ""
        return f"{prefix}{quote}{REDACTED_SECRET}{quote}"

    return _SECRET_PATTERN.sub(replace, text), replacements


def redact_message_content(content: Any) -> Any:
    """Redact strings and text blocks without changing non-text message content."""
    if isinstance(content, str):
        redacted, replacements = redact_secrets(content)
        return redacted if replacements else content
    if not isinstance(content, list):
        return content

    changed = False
    blocks: list[Any] = []
    for block in content:
        if isinstance(block, str):
            redacted = redact_message_content(block)
            changed = changed or redacted is not block
            blocks.append(redacted)
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            redacted = redact_message_content(block["text"])
            changed = changed or redacted is not block["text"]
            blocks.append({**block, "text": redacted})
        else:
            blocks.append(block)
    return blocks if changed else content


def create_secret_anonymizer() -> Callable[[Any], Any]:
    """Build a recursive trace anonymizer using the same placeholder-aware redactor."""
    return create_anonymizer(lambda text, _path: redact_secrets(text)[0])
