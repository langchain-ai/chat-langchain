"""Tell the model which language the user wrote in, on every model call.

The prompt already says to answer in the user's language, but short questions
written in Chinese/Japanese/Korean/Arabic/Cyrillic around an English product
term ("graph.astream 如何用", "entity 相关的内容") still get English replies on
about 3% of non-Latin-script questions (real prod, Nov 13-16). The script of a
message is easy to detect in code, so this middleware detects it and adds a
one-line reply-language note to the latest user message for the model call
only (state and the stored conversation are not changed). The note sits after
the static system prompt, so prompt caching is unaffected.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import HumanMessage

# Checked in order: kana before CJK so Japanese with kanji reads as Japanese.
_SCRIPTS = [
    ("Japanese", re.compile(r"[\u3040-\u30ff]")),
    ("Korean", re.compile(r"[\uac00-\ud7af]")),
    ("Chinese", re.compile(r"[\u4e00-\u9fff]")),
    ("the Arabic-script language the user wrote in (Arabic, Persian, Urdu...)", re.compile(r"[\u0600-\u06ff]")),
    ("the Cyrillic-script language the user wrote in (Russian, Ukrainian...)", re.compile(r"[\u0400-\u04ff]")),
    ("Hindi", re.compile(r"[\u0900-\u097f]")),
    ("Thai", re.compile(r"[\u0e00-\u0e7f]")),
]
_NOTE_SPLIT = re.compile(r"\s---\s")
_CODE = re.compile(r"```.*?```|`[^`]*`|https?://\S+", re.S)
_UNIT = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]|[^\W\d_\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+")
MARKER = "[Reply language:"


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def detect_reply_language(text: str) -> str | None:
    """Return a language name when the message is mainly in a non-Latin script."""
    body = _CODE.sub(" ", _NOTE_SPLIT.split(text)[0])
    units = _UNIT.findall(body)
    if not units:
        return None
    total = len(units)
    for name, pattern in _SCRIPTS:
        hits = sum(1 for u in units if pattern.search(u))
        if hits / total > 0.15:
            return name
    return None


def _note(language: str) -> str:
    return (
        f"\n\n{MARKER} the user wrote this message in {language}. Write the whole reply, "
        f"starting with the bold first sentence, in {language}; keep code, identifiers and URLs as they are.]"
    )


def add_language_note(messages: list) -> list:
    """Return messages with the note appended to the last user message, if needed."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if not isinstance(msg, HumanMessage):
            continue
        text = _text(msg.content)
        if MARKER in text:
            return messages
        language = detect_reply_language(text)
        if not language:
            return messages
        content = msg.content
        if isinstance(content, str):
            content = content + _note(language)
        else:
            content = list(content) + [{"type": "text", "text": _note(language)}]
        out = list(messages)
        out[i] = msg.model_copy(update={"content": content})
        return out
    return messages


class ReplyLanguageMiddleware(AgentMiddleware):
    """Add a per-call reply-language note for non-Latin-script user messages."""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        messages = add_language_note(request.messages)
        if messages is not request.messages:
            request = request.override(messages=messages)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        messages = add_language_note(request.messages)
        if messages is not request.messages:
            request = request.override(messages=messages)
        return await handler(request)
