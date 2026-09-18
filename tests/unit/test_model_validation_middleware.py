"""Tests for the non-existent model ID validation guard."""

from __future__ import annotations

import os
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.model_validation_middleware import ModelValidationMiddleware
from src.utils.model_id_validation import (
    SUPPORTED_MODEL_SUBSTITUTE,
    contains_invalid_model_id,
    substitute_invalid_model_ids,
)


def test_helper_detects_and_substitutes_invalid_ids():
    text = 'llm = init_chat_model(model="gpt-5.5")'
    assert contains_invalid_model_id(text)
    repaired = substitute_invalid_model_ids(text)
    assert "gpt-5.5" not in repaired
    assert SUPPORTED_MODEL_SUBSTITUTE in repaired


def test_helper_ignores_valid_ids():
    assert not contains_invalid_model_id('model="gpt-4o"')


def test_after_model_rewrites_invalid_model_id():
    middleware = ModelValidationMiddleware()
    answer = AIMessage(content='Use `model="gpt-5.5"` here.', id="a1")
    state = {"messages": [HumanMessage(content="how?"), answer]}

    update = middleware.after_model(state, runtime=SimpleNamespace())

    assert update is not None
    content = update["messages"][0].content
    assert "gpt-5.5" not in content
    assert SUPPORTED_MODEL_SUBSTITUTE in content


def test_after_model_noop_for_valid_answer():
    middleware = ModelValidationMiddleware()
    answer = AIMessage(content='Use `model="gpt-4o"`.', id="a1")
    state = {"messages": [answer]}

    assert middleware.after_model(state, runtime=SimpleNamespace()) is None


def test_after_model_skips_tool_call_messages():
    middleware = ModelValidationMiddleware()
    answer = AIMessage(
        content='model="gpt-5.5"',
        id="a1",
        tool_calls=[{"name": "search", "args": {}, "id": "t1"}],
    )
    state = {"messages": [answer]}

    assert middleware.after_model(state, runtime=SimpleNamespace()) is None
