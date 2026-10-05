from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.provider_prefill_middleware import ProviderPrefillMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


def _request(model, messages):
    return ModelRequest(model=model, messages=messages)


def test_google_requests_drop_trailing_assistant_prefill():
    middleware = ProviderPrefillMiddleware()
    request = _request(
        SimpleNamespace(_llm_type="google_generative_ai"),
        [HumanMessage(content="Question"), AIMessage(content="prefill")],
    )
    captured = []

    middleware.wrap_model_call(request, lambda normalized: captured.append(normalized))

    assert [message.type for message in captured[0].messages] == ["human"]


def test_google_requests_preserve_function_responses():
    middleware = ProviderPrefillMiddleware()
    request = _request(
        SimpleNamespace(_llm_type="google_generative_ai"),
        [
            HumanMessage(content="Question"),
            AIMessage(
                content="", tool_calls=[{"name": "lookup", "args": {}, "id": "1"}]
            ),
            ToolMessage(content="Result", tool_call_id="1"),
        ],
    )
    captured = []

    middleware.wrap_model_call(request, lambda normalized: captured.append(normalized))

    assert captured[0].messages is request.messages


def test_non_google_requests_preserve_assistant_prefill():
    middleware = ProviderPrefillMiddleware()
    request = _request(
        SimpleNamespace(_llm_type="openai"),
        [HumanMessage(content="Question"), AIMessage(content="prefill")],
    )
    captured = []

    middleware.wrap_model_call(request, lambda normalized: captured.append(normalized))

    assert captured[0].messages is request.messages


@pytest.mark.anyio
async def test_prefill_value_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2)
    attempts = 0

    async def handler(_request):
        nonlocal attempts
        attempts += 1
        raise ValueError("does not support model prefilling")

    with pytest.raises(ValueError, match="does not support model prefilling"):
        await middleware.awrap_model_call(object(), handler)

    assert attempts == 1
