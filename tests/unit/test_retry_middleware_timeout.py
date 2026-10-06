import asyncio

import pytest

from src.middleware.retry_middleware import ModelRetryMiddleware


class HangingChatModel:
    async def ainvoke(self, _input):
        await asyncio.Event().wait()


def test_hanging_model_call_times_out():
    model = HangingChatModel()
    middleware = ModelRetryMiddleware(max_retries=0, timeout=0.01)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(middleware.awrap_model_call(object(), model.ainvoke))
