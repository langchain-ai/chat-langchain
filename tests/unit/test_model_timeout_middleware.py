"""Tests for model timeout and retry middleware composition."""

import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest

from src.middleware.model_timeout_middleware import ModelCallTimeoutMiddleware
from src.middleware.retry_middleware import ModelRetryMiddleware


def test_stalled_model_call_times_out_and_retries():
    """A stalled model call raises after the configured retry budget."""

    async def run_test():
        calls = 0
        timeout_middleware = ModelCallTimeoutMiddleware(timeout=0.01)
        retry_middleware = ModelRetryMiddleware(max_retries=1, initial_delay=0)
        request = ModelRequest(model=object(), messages=[])

        async def stalled_handler(_request):
            nonlocal calls
            calls += 1
            await asyncio.Event().wait()

        async def retry_handler(retry_request):
            return await retry_middleware.awrap_model_call(
                retry_request,
                lambda timeout_request: timeout_middleware.awrap_model_call(
                    timeout_request, stalled_handler
                ),
            )

        with pytest.raises(TimeoutError):
            await asyncio.wait_for(retry_handler(request), timeout=1)

        assert calls == 2

    asyncio.run(run_test())
