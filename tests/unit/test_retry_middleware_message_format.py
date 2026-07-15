"""Tests that ModelRetryMiddleware pins output_version for span metadata."""

from __future__ import annotations

import dataclasses

from langchain.agents.middleware.types import ModelRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


def _make_request(model) -> ModelRequest:
    fields = {f.name: None for f in dataclasses.fields(ModelRequest)}
    fields["model"] = model
    fields["messages"] = []
    return ModelRequest(**fields)


def test_ensure_message_format_pins_output_version():
    model = GenericFakeChatModel(messages=iter([AIMessage("hi")]))
    assert model.output_version != "v1"

    middleware = ModelRetryMiddleware(max_retries=0)
    updated = middleware._ensure_message_format(_make_request(model))

    assert updated.model.output_version == "v1"
