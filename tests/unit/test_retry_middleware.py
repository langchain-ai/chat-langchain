import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_retry_middleware_times_out_stalled_model_call(caplog):
    asyncio.run(_assert_stalled_model_call_times_out(caplog))


async def _assert_stalled_model_call_times_out(caplog):
    middleware = ModelRetryMiddleware(
        max_retries=1,
        initial_delay=0,
        timeout=0.01,
    )
    request = ModelRequest(model=object(), messages=[])
    never_set = asyncio.Event()

    async def stalled_handler(_request):
        await never_set.wait()

    with pytest.raises(TimeoutError):
        await middleware.awrap_model_call(request, stalled_handler)

    assert "Model call timed out after 0.01s" in caplog.text
