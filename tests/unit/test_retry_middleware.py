"""Tests for bounded model calls and unusable response retries."""

import asyncio
import random
import string
import zlib
from unittest.mock import Mock

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import Field

from src.middleware import retry_middleware
from src.middleware.retry_middleware import (
    MalformedResponseError,
    ModelRetryMiddleware,
    _is_degenerate_output,
    get_retryable_response_reason,
)


@pytest.mark.parametrize("length", [0, 3999, 4000, 4001, 8000])
def test_repetitive_text_threshold(length):
    assert _is_degenerate_output("*" * length) is (length > 4000)


@pytest.mark.parametrize(
    "compressed_length, expected", [(249, True), (250, False), (251, False)]
)
def test_compression_ratio_boundary(monkeypatch, compressed_length, expected):
    monkeypatch.setattr(
        retry_middleware.zlib, "compress", lambda raw: b"x" * compressed_length
    )
    assert _is_degenerate_output("*" * 5000) is expected


@pytest.mark.parametrize(
    "noise_length, expected", [(100, True), (300, False), (4001, False)]
)
def test_real_compression_on_both_sides_of_ratio(noise_length, expected):
    random_generator = random.Random(42)
    noise = "".join(random_generator.choices(string.ascii_letters, k=noise_length))
    text = noise + "*" * (4001 - noise_length)
    assert (len(zlib.compress(text.encode())) / len(text.encode()) < 0.05) is expected
    assert _is_degenerate_output(text) is expected


def test_compression_ratio_uses_utf8_bytes(monkeypatch):
    compress = Mock(return_value=b"x" * 300)
    monkeypatch.setattr(retry_middleware.zlib, "compress", compress)
    assert _is_degenerate_output("é" * 4001)
    compress.assert_called_once_with(("é" * 4001).encode("utf-8"))


@pytest.mark.parametrize(
    "finish_reason", ["MAX_TOKENS", "length", "MALFORMED_FUNCTION_CALL"]
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_retryable_finish_reasons(finish_reason, wrapped):
    message = AIMessage(
        content="partial answer", response_metadata={"finish_reason": finish_reason}
    )
    response = ModelResponse(result=[message]) if wrapped else message
    assert get_retryable_response_reason(response) == finish_reason


@pytest.mark.parametrize("finish_reason", ["STOP", "stop", "tool_calls", "SAFETY", ""])
def test_unrelated_finish_reasons_are_unchanged(finish_reason):
    message = AIMessage(
        content="ordinary answer", response_metadata={"finish_reason": finish_reason}
    )
    assert get_retryable_response_reason(ModelResponse(result=[message])) is None


def test_repetitive_content_blocks_with_stop_are_retryable():
    message = AIMessage(
        content=[{"type": "text", "text": "*   *   *\n" * 1000}],
        response_metadata={"finish_reason": "STOP"},
    )
    assert (
        get_retryable_response_reason(ModelResponse(result=[message]))
        == "degenerate output"
    )


def _call(middleware, request, handler, asynchronous):
    if not asynchronous:
        return middleware.wrap_model_call(request, handler)

    async def async_handler(bounded_request):
        return handler(bounded_request)

    return asyncio.run(middleware.awrap_model_call(request, async_handler))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("max_retries", [0, 2])
@pytest.mark.parametrize(
    "reason", ["MAX_TOKENS", "length", "MALFORMED_FUNCTION_CALL", "repetition"]
)
def test_exhausted_retries_raise_malformed_response(asynchronous, max_retries, reason):
    message = AIMessage(
        content="*" * 5000 if reason == "repetition" else "partial answer",
        response_metadata={
            "finish_reason": "STOP" if reason == "repetition" else reason
        },
    )
    model = FakeMessagesListChatModel(responses=[message])
    request = ModelRequest(model=model, messages=[])
    handler = Mock(return_value=ModelResponse(result=[message]))
    middleware = ModelRetryMiddleware(max_retries=max_retries, initial_delay=0)
    with pytest.raises(MalformedResponseError):
        _call(middleware, request, handler, asynchronous)
    assert handler.call_count == max_retries + 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_retry_can_recover_without_fallback(asynchronous):
    ordinary = ModelResponse(result=[AIMessage(content="ordinary answer")])
    handler = Mock(
        side_effect=[ModelResponse(result=[AIMessage(content="*" * 5000)]), ordinary]
    )
    request = ModelRequest(model=FakeMessagesListChatModel(responses=[]), messages=[])
    middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0)
    assert _call(middleware, request, handler, asynchronous) is ordinary
    assert handler.call_count == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("google", [False, True])
def test_request_output_cap_and_ordinary_response(asynchronous, google):
    model = (
        ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", api_key="test")
        if google
        else FakeMessagesListChatModel(responses=[])
    )
    setting = "max_output_tokens" if google else "max_tokens"
    request = ModelRequest(
        model=model, messages=[], model_settings={"temperature": 0.2, setting: 99999}
    )
    response = ModelResponse(result=[AIMessage(content="ordinary answer")])
    handler = Mock(return_value=response)
    middleware = ModelRetryMiddleware(max_output_tokens=1234)
    assert _call(middleware, request, handler, asynchronous) is response
    handler.assert_called_once()
    bounded_request = handler.call_args.args[0]
    assert bounded_request.model_settings == {"temperature": 0.2, setting: 1234}
    assert request.model_settings == {"temperature": 0.2, setting: 99999}
    if google:
        provider_request = model._prepare_request(
            [HumanMessage(content="hello")], **bounded_request.model_settings
        )
        assert provider_request["config"].max_output_tokens == 1234


class RecordingGoogleModel(ChatGoogleGenerativeAI):
    responses: list[AIMessage]
    calls: list[dict] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(kwargs)
        return ChatResult(generations=[ChatGeneration(message=self.responses[0])])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class RecordingFallbackModel(FakeMessagesListChatModel):
    calls: list[dict] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(kwargs)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reason", ["MAX_TOKENS", "length", "repetition", "STOP"])
def test_agent_retries_then_falls_back_without_storing_bad_output(asynchronous, reason):
    primary_message = AIMessage(
        content="*" * 5000 if reason == "repetition" else "ordinary answer",
        response_metadata={
            "finish_reason": "STOP" if reason == "repetition" else reason
        },
    )
    primary = RecordingGoogleModel(
        model="gemini-3.5-flash-lite", api_key="test", responses=[primary_message]
    )
    fallback = RecordingFallbackModel(responses=[AIMessage(content="fallback answer")])
    middleware = [
        ModelFallbackMiddleware(fallback),
        ModelRetryMiddleware(max_retries=1, initial_delay=0, max_output_tokens=1234),
    ]
    graph = create_agent(model=primary, middleware=middleware)
    inputs = {"messages": [HumanMessage(content="How do agents work?")]}
    result = (
        asyncio.run(graph.ainvoke(inputs)) if asynchronous else graph.invoke(inputs)
    )
    retryable = reason != "STOP"
    assert result["messages"][-1].content == (
        "fallback answer" if retryable else "ordinary answer"
    )
    assert len(result["messages"]) == 2
    assert len(primary.calls) == (2 if retryable else 1)
    assert len(fallback.calls) == (1 if retryable else 0)
    assert all(call["max_output_tokens"] == 1234 for call in primary.calls)
    assert all(call["max_tokens"] == 1234 for call in fallback.calls)
