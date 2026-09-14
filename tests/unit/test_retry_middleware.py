import asyncio

import pytest
from langchain.agents.middleware.types import ModelRequest

from src.middleware.retry_middleware import ModelRetryMiddleware


def test_model_prefilling_error_is_not_retried():
    middleware = ModelRetryMiddleware(max_retries=2)
    calls = 0

    async def handler(request: ModelRequest):  # noqa: ARG001
        nonlocal calls
        calls += 1
        raise ValueError("model does not support model prefilling")

    with pytest.raises(ValueError, match="does not support model prefilling"):
        asyncio.run(
            middleware.awrap_model_call(
                ModelRequest(model=object(), messages=[]), handler
            )
        )

    assert calls == 1
