import asyncio

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_model_call_timeout_retries_then_raises():
    async def run_test():
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            await asyncio.sleep(1)

        middleware = ModelRetryMiddleware(
            max_retries=1,
            initial_delay=0,
            timeout_s=0.001,
        )

        with pytest.raises(TimeoutError):
            await middleware.awrap_model_call(None, handler)

        assert calls == 2

    asyncio.run(run_test())
