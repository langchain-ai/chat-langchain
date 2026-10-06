import asyncio
import time

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_model_retry_middleware_times_out_hung_handler():
    middleware = ModelRetryMiddleware(
        max_retries=2,
        initial_delay=0,
        timeout_seconds=0.05,
    )
    calls = 0
    never_set = asyncio.Event()

    async def handler(request):  # noqa: ARG001
        nonlocal calls
        calls += 1
        await never_set.wait()

    started_at = time.monotonic()
    with pytest.raises(TimeoutError):
        asyncio.run(middleware.awrap_model_call(None, handler))
    elapsed = time.monotonic() - started_at

    assert calls == 3
    assert elapsed < 0.5
