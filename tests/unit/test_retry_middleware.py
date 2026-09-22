import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.retry_middleware import (
    ModelRetryMiddleware,
    _ProviderValidationAwareRunnableRetry,
)
from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def test_provider_validation_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    calls = 0

    async def handler(request: ModelRequest):
        nonlocal calls
        calls += 1
        raise ValueError("unsupported request shape")

    with pytest.raises(ValueError, match="unsupported request shape"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[HumanMessage(content="Hi")]),
                handler,
            )
        )

    assert calls == 1


def test_model_retry_wrapper_does_not_retry_provider_validation_error():
    calls = 0

    def invoke(_input):
        nonlocal calls
        calls += 1
        raise ValueError("unsupported request shape")

    runnable = _ProviderValidationAwareRunnableRetry(
        bound=RunnableLambda(invoke), max_attempt_number=3
    )

    with pytest.raises(ValueError, match="unsupported request shape"):
        runnable.invoke("request")

    assert calls == 1


def test_tool_validation_error_is_not_retried():
    middleware = ToolRetryMiddleware(max_attempts=3, initial_delay=0)
    request = ToolCallRequest(
        tool_call={"name": "check_links", "id": "call-1"},
        tool=None,
        state=None,
        runtime=None,
    )
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        raise ValueError("2 validation errors for CheckLinksInput\nurls")

    result = asyncio.run(middleware.awrap_tool_call(request, handler))

    assert calls == 1
    assert result.status == "error"
    assert "Expected urls as a URL, a list of URLs" in result.content
    assert "validation errors" not in result.content
