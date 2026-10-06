import asyncio
from typing import cast

import pytest
from langchain.agents.middleware.types import ModelRequest, ModelResponse

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_timeout_retries_until_exhausted():
    async def run_test():
        middleware = ModelRetryMiddleware(
            max_retries=2,
            initial_delay=0,
            timeout=0.01,
        )
        pending = asyncio.Event()
        calls = 0

        async def handler(request: ModelRequest) -> ModelResponse:
            nonlocal calls
            calls += 1
            await pending.wait()
            raise AssertionError("unreachable")

        with pytest.raises(TimeoutError):
            await middleware.awrap_model_call(cast(ModelRequest, object()), handler)

        assert calls == 3

    asyncio.run(run_test())


def test_fast_handler_returns_normally():
    async def run_test():
        middleware = ModelRetryMiddleware(timeout=0.1)
        response = cast(ModelResponse, object())

        async def handler(request: ModelRequest) -> ModelResponse:
            return response

        result = await middleware.awrap_model_call(
            cast(ModelRequest, object()), handler
        )

        assert result is response

    asyncio.run(run_test())
