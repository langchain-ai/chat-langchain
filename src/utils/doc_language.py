"""Infer the documentation language from the conversation."""

import re
from collections.abc import Iterable
from typing import Any, Literal

from langchain_core.messages import HumanMessage

DocLanguage = Literal["python", "javascript"]

_LANGUAGE_MENTIONS = {
    "python": re.compile(r"\bpython\b", re.IGNORECASE),
    "javascript": re.compile(r"\b(?:javascript|typescript|js|ts)\b", re.IGNORECASE),
}
_PYTHON_CODE = re.compile(
    r"```\s*(?:python|py)\b|\b(?:def|async\s+def)\s+\w+\s*\(|"
    r"\b(?:from\s+\w+(?:\.\w+)*\s+import|import\s+\w+)\b|"
    r"\b(?:PostgresSaver|MessagesState)\b"
)
_JAVASCRIPT_CODE = re.compile(
    r"```\s*(?:javascript|js|typescript|ts)\b|\b(?:useStream|const)\b|"
    r"\b(?:function|interface)\s+\w+|=>|\b(?:import\s+\{|require\s*\()"
)


def _message_text(message: Any) -> str:
    content = (
        message.get("content", "") if isinstance(message, dict) else message.content
    )
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def _is_user_message(message: Any) -> bool:
    if isinstance(message, HumanMessage):
        return True
    if isinstance(message, dict):
        return message.get("role") in {"user", "human"}
    return False


def _language_from_text(text: str) -> DocLanguage | None:
    python_evidence = bool(
        _LANGUAGE_MENTIONS["python"].search(text) or _PYTHON_CODE.search(text)
    )
    javascript_evidence = bool(
        _LANGUAGE_MENTIONS["javascript"].search(text) or _JAVASCRIPT_CODE.search(text)
    )
    if python_evidence == javascript_evidence:
        return None
    return "python" if python_evidence else "javascript"


def infer_doc_language(messages: Iterable[Any]) -> DocLanguage | None:
    """Infer an unambiguous documentation language from user turns."""
    user_messages = [message for message in messages if _is_user_message(message)]
    if not user_messages:
        return None

    current_language = _language_from_text(_message_text(user_messages[-1]))
    current_has_evidence = bool(
        _LANGUAGE_MENTIONS["python"].search(_message_text(user_messages[-1]))
        or _LANGUAGE_MENTIONS["javascript"].search(_message_text(user_messages[-1]))
        or _PYTHON_CODE.search(_message_text(user_messages[-1]))
        or _JAVASCRIPT_CODE.search(_message_text(user_messages[-1]))
    )
    if current_has_evidence:
        return current_language

    for message in reversed(user_messages[:-1]):
        language = _language_from_text(_message_text(message))
        if language is not None:
            return language
    return None


__all__ = ["DocLanguage", "infer_doc_language"]
