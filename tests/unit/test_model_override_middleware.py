from unittest.mock import MagicMock

import pytest

from src.middleware.model_override_middleware import ModelOverrideMiddleware


def test_overrides_model_sync():
    model = object()
    request = MagicMock()
    request.override.return_value = "overridden"
    handler = MagicMock(return_value="response")
    out = ModelOverrideMiddleware(model).wrap_model_call(request, handler)
    request.override.assert_called_once_with(model=model)
    handler.assert_called_once_with("overridden")
    assert out == "response"


@pytest.mark.asyncio
async def test_overrides_model_async():
    model = object()
    request = MagicMock()
    request.override.return_value = "overridden"

    async def handler(req):
        return ("response", req)

    out = await ModelOverrideMiddleware(model).awrap_model_call(request, handler)
    request.override.assert_called_once_with(model=model)
    assert out == ("response", "overridden")
