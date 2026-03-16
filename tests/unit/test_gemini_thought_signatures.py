"""Tests for Gemini thought signature leaking into output content.

Bug: GuardrailsMiddleware._generate_rejection_message() does
``return AIMessage(content=response.content)``.  When the Gemini LLM returns
a response whose ``.content`` is a list of dicts (e.g.
``[{'type': 'text', 'text': '...', 'extras': {'signature': '<base64>'}}]``),
that list is forwarded verbatim as the AIMessage content.  Clients then receive
a list object instead of a plain string, requiring extra parsing to extract the
answer.

Root cause: src/middleware/guardrails_middleware.py line 174 — the raw
``response.content`` (which may be a list when Gemini attaches thought
signatures) is passed directly to ``AIMessage(content=...)``.

Fix: extract the plain text from ``response.content`` before constructing the
final AIMessage.
"""

import asyncio
from unittest.mock import AsyncMock

from langchain_core.messages import AIMessage

from src.middleware.guardrails_middleware import GuardrailsMiddleware


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_gemini_ai_message(text: str, signature: str = "dGVzdA==") -> AIMessage:
    """Return an AIMessage that looks like a Gemini response with a thought signature.

    The ``extras.signature`` field contains a base64-encoded string, exactly as
    langchain-google-genai produces when ``thought_signature`` is present on a
    response Part.
    """
    content_list = [
        {
            "type": "text",
            "text": text,
            "extras": {"signature": signature},
            "index": 0,
        }
    ]
    return AIMessage(content=content_list)


def _make_plain_ai_message(text: str) -> AIMessage:
    """Return an AIMessage with a plain string content (non-Gemini path)."""
    return AIMessage(content=text)


def _build_middleware() -> GuardrailsMiddleware:
    """Create a GuardrailsMiddleware instance without calling __init__."""
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.llm = AsyncMock()
    return middleware


# ---------------------------------------------------------------------------
# 1.  Output content must always be a plain string — never a list
# ---------------------------------------------------------------------------


def test_rejection_message_content_is_string_when_gemini_returns_list():
    """_generate_rejection_message must return a plain-string AIMessage even when
    the underlying LLM returns a list-type content (Gemini thought signatures).
    """
    gemini_response = _make_gemini_ai_message("I can only help with LangChain topics.")

    middleware = _build_middleware()
    middleware.llm.ainvoke = AsyncMock(return_value=gemini_response)

    result = asyncio.get_event_loop().run_until_complete(
        middleware._generate_rejection_message("Tell me a joke")
    )

    assert isinstance(result.content, str), (
        f"Expected str content, got {type(result.content)}: {result.content!r}\n"
        "Gemini thought signatures are leaking into the output AIMessage content."
    )


def test_rejection_message_content_does_not_contain_extras_signature():
    """Output AIMessage.content must not contain dicts with 'extras.signature' keys."""
    gemini_response = _make_gemini_ai_message("I can only help with LangChain topics.")

    middleware = _build_middleware()
    middleware.llm.ainvoke = AsyncMock(return_value=gemini_response)

    result = asyncio.get_event_loop().run_until_complete(
        middleware._generate_rejection_message("Tell me a joke")
    )

    # If content is a list, check for signature leakage
    if isinstance(result.content, list):
        for item in result.content:
            if isinstance(item, dict):
                extras = item.get("extras", {})
                assert "signature" not in extras, (
                    f"Gemini thought signature leaked into output: {item!r}"
                )


def test_rejection_message_text_is_preserved():
    """The plain-text answer from the LLM must be preserved in the output."""
    expected_text = "I can only help with LangChain topics."
    gemini_response = _make_gemini_ai_message(expected_text)

    middleware = _build_middleware()
    middleware.llm.ainvoke = AsyncMock(return_value=gemini_response)

    result = asyncio.get_event_loop().run_until_complete(
        middleware._generate_rejection_message("Tell me a joke")
    )

    assert expected_text in result.content, (
        f"Expected text '{expected_text}' not found in result.content={result.content!r}"
    )


# ---------------------------------------------------------------------------
# 2.  Plain-string (non-Gemini) responses must continue to work
# ---------------------------------------------------------------------------


def test_rejection_message_plain_string_response_unchanged():
    """When the LLM returns a plain-string AIMessage the content must be preserved."""
    plain_response = _make_plain_ai_message("I can only help with LangChain topics.")

    middleware = _build_middleware()
    middleware.llm.ainvoke = AsyncMock(return_value=plain_response)

    result = asyncio.get_event_loop().run_until_complete(
        middleware._generate_rejection_message("Tell me a joke")
    )

    assert isinstance(result.content, str)
    assert result.content == "I can only help with LangChain topics."


# ---------------------------------------------------------------------------
# 3.  Multi-block list content: all text blocks must be joined
# ---------------------------------------------------------------------------


def test_rejection_message_multi_block_content_joined():
    """Multi-block list content (thinking + text) must be flattened to a single string."""
    multi_block_response = AIMessage(
        content=[
            {"type": "thinking", "thinking": "Let me think...", "signature": "dGVzdA=="},
            {
                "type": "text",
                "text": "I can only help with LangChain topics.",
                "extras": {"signature": "dGVzdA=="},
            },
        ]
    )

    middleware = _build_middleware()
    middleware.llm.ainvoke = AsyncMock(return_value=multi_block_response)

    result = asyncio.get_event_loop().run_until_complete(
        middleware._generate_rejection_message("Tell me a joke")
    )

    assert isinstance(result.content, str), (
        f"Expected str, got {type(result.content)}: {result.content!r}"
    )
    assert "LangChain" in result.content


# ---------------------------------------------------------------------------
# 4.  Fallback on LLM error must also be a plain string
# ---------------------------------------------------------------------------


def test_rejection_message_fallback_on_llm_error_is_string():
    """When LLM raises an exception the fallback message must be a plain string."""
    middleware = _build_middleware()
    middleware.llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))

    result = asyncio.get_event_loop().run_until_complete(
        middleware._generate_rejection_message("Tell me a joke")
    )

    assert isinstance(result.content, str), (
        f"Fallback content must be a str, got {type(result.content)}: {result.content!r}"
    )
    assert len(result.content) > 0
