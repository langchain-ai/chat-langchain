"""Tests for request error visibility inside the asynchronous fallback chain."""

import asyncio
import logging

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.primary_model_request_error_middleware import (
    PrimaryModelRequestErrorMiddleware,
)

PRIMARY_MODEL = "google_genai:gemini-3.5-flash-lite"


class FailingChatModel(FakeMessagesListChatModel):
    model: str
    error: Exception | None = None

    def _generate(self, *args, **kwargs):
        if self.error is not None:
            raise self.error
        return super()._generate(*args, **kwargs)


def _invoke_with_fallback(primary_error, fallback_error=None):
    primary = FailingChatModel(
        model="gemini-3.5-flash-lite",
        responses=[AIMessage(content="primary answer")],
        error=primary_error,
    )
    fallback = FailingChatModel(
        model="gpt-5.4-nano",
        responses=[AIMessage(content="fallback answer")],
        error=fallback_error,
    )
    final_fallback = FakeMessagesListChatModel(
        responses=[AIMessage(content="final fallback answer")]
    )
    agent = create_agent(
        model=primary,
        middleware=[
            ModelFallbackMiddleware(fallback, final_fallback),
            PrimaryModelRequestErrorMiddleware(PRIMARY_MODEL),
        ],
    )
    return asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="Hi")]}))


@pytest.mark.parametrize("exception_type", [ValueError, TypeError])
def test_primary_request_error_is_logged_before_successful_fallback(
    caplog, exception_type
):
    message = "Unrecognized tool choice format {'type': 'function'}"

    with caplog.at_level(logging.ERROR):
        result = _invoke_with_fallback(exception_type(message))

    assert result["messages"][-1].content == "fallback answer"
    errors = [
        record
        for record in caplog.records
        if "primary_model_request_error" in record.getMessage()
    ]
    assert len(errors) == 1
    assert errors[0].levelno == logging.ERROR
    assert PRIMARY_MODEL in errors[0].getMessage()
    assert message in errors[0].getMessage()


@pytest.mark.parametrize(
    "error",
    [TimeoutError("timed out"), RuntimeError("rate limited"), RuntimeError("HTTP 503")],
)
def test_transient_primary_failure_still_falls_back_without_error_marker(caplog, error):
    with caplog.at_level(logging.ERROR):
        result = _invoke_with_fallback(error)

    assert result["messages"][-1].content == "fallback answer"
    assert "primary_model_request_error" not in caplog.text


def test_fallback_request_error_is_not_reported_as_primary_error(caplog):
    with caplog.at_level(logging.ERROR):
        result = _invoke_with_fallback(
            RuntimeError("primary unavailable"), ValueError("invalid fallback request")
        )

    assert result["messages"][-1].content == "final fallback answer"
    assert "primary_model_request_error" not in caplog.text


def test_successful_primary_call_does_not_emit_error_marker(caplog):
    with caplog.at_level(logging.ERROR):
        result = _invoke_with_fallback(None)

    assert result["messages"][-1].content == "primary answer"
    assert "primary_model_request_error" not in caplog.text
