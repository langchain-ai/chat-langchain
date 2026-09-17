import asyncio
from unittest.mock import MagicMock

import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import HumanMessage

from src.middleware.retry_middleware import ModelRetryMiddleware


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Unrecognized tool choice format"),
        ValueError("Invalid tool schema"),
    ],
)
def test_retry_middleware_reraises_request_shape_value_errors(error):
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    request = ModelRequest(model=MagicMock(), messages=[HumanMessage(content="query")])
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(ValueError, match=str(error)):
        asyncio.run(middleware.awrap_model_call(request, handler))

    assert calls == 1


class InvalidRequestError(Exception):
    status_code = 400
    type = "invalid_request_error"


def test_retry_middleware_reraises_provider_invalid_request_without_retry():
    middleware = ModelRetryMiddleware(max_retries=2, initial_delay=0)
    request = ModelRequest(model=MagicMock(), messages=[HumanMessage(content="query")])
    calls = 0

    async def handler(_request):
        nonlocal calls
        calls += 1
        raise InvalidRequestError("tool_choice is invalid")

    with pytest.raises(InvalidRequestError):
        asyncio.run(middleware.awrap_model_call(request, handler))

    assert calls == 1
