"""Redact credential values from user-provided text."""

from __future__ import annotations

import re

_REDACTION = "<REDACTED>"
_URI_CREDENTIAL_RE = re.compile(
    r"(?P<prefix>(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*)://"
    r"(?P<user>[^:/\s@]+):)(?P<value>[^@\s/]+)(?=@)"
)
_PREFIX_SECRET_RE = re.compile(
    r"(?P<prefix>sk-|tvly-|AIza|ghp_|xoxb-|pk_live_|lsv2_|lcl_|AKIA)"
    r"(?P<value>[^\s\"'`,;)}\]]+)"
)
_BEARER_SECRET_RE = re.compile(
    r"(?P<prefix>Bearer\s+)(?P<value>[^\s\"'`,;)}\]]+)",
    re.IGNORECASE,
)
_ASSIGNMENT_RE = re.compile(
    r"(?P<prefix>\b(?:api[_-]?key|token|secret|password|passwd|pwd)"
    r"\s*[:=]\s*)"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    re.IGNORECASE,
)


def _is_placeholder(value: str) -> bool:
    if len(value) < 8:
        return True
    if re.fullmatch(r"YOUR_[A-Za-z0-9_*.-]+", value):
        return True
    if re.fullmatch(r"<[^>]+>|\$\{[^}]+\}|os\.getenv\([^)]*\)", value):
        return True
    if value.lower() in {"changeme", "example"}:
        return True
    return re.fullmatch(r"x+", value, re.IGNORECASE) is not None


def _replace_value(match: re.Match[str]) -> str:
    value = match.group("value")
    if _REDACTION in value:
        return match.group(0)
    if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
        quote = value[0]
        inner_value = value[1:-1]
        if _is_placeholder(inner_value):
            return match.group(0)
        return f"{match.group('prefix')}{quote}{_REDACTION}{quote}"
    if _is_placeholder(value):
        return match.group(0)
    return f"{match.group('prefix')}{_REDACTION}"


def redact_secrets(text: str) -> str:
    """Replace detected secret values with a redaction marker."""
    if not isinstance(text, str):
        return text
    redacted = _URI_CREDENTIAL_RE.sub(_replace_value, text)
    redacted = _ASSIGNMENT_RE.sub(_replace_value, redacted)
    redacted = _PREFIX_SECRET_RE.sub(_replace_value, redacted)
    return _BEARER_SECRET_RE.sub(_replace_value, redacted)


__all__ = ["redact_secrets"]
