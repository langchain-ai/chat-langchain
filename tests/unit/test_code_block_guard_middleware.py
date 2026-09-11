"""Tests for executable code block validation."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage


def _request() -> ModelRequest:
    return ModelRequest(model=object(), messages=[HumanMessage(content="Give me code")])


def _response(content: str) -> ModelResponse:
    return ModelResponse(result=[AIMessage(content=content)])


def test_valid_python_block_passes_untouched():
    from src.middleware.code_block_guard_middleware import CodeBlockGuardMiddleware

    middleware = CodeBlockGuardMiddleware()
    response = _response("```python\nvalue = 1\n```")

    async def handler(request: ModelRequest) -> ModelResponse:
        return response

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert result is response
    assert response.result[0].content == "```python\nvalue = 1\n```"


def test_anonymous_async_def_triggers_exactly_one_retry():
    from src.middleware.code_block_guard_middleware import CodeBlockGuardMiddleware

    middleware = CodeBlockGuardMiddleware()
    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return _response(
                "```python\nregister(async def callback():\n    pass)\n```"
            )
        return _response("```python\nasync def callback():\n    pass\n```")

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert len(calls) == 2
    assert "invalid syntax" in calls[1].messages[-1].content
    assert isinstance(calls[1].messages[-1], HumanMessage)
    assert "async def callback" in result.result[0].content


def test_invalid_virtualenv_activation_triggers_exactly_one_retry():
    from src.middleware.code_block_guard_middleware import CodeBlockGuardMiddleware

    middleware = CodeBlockGuardMiddleware()
    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return _response("```bash\nsource .venv/activate\n```")
        return _response("```bash\nsource .venv/bin/activate\n```")

    result = asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert len(calls) == 2
    assert "invalid virtualenv activation path" in calls[1].messages[-1].content
    assert isinstance(calls[1].messages[-1], HumanMessage)
    assert "source .venv/bin/activate" in result.result[0].content


def test_retry_request_ends_with_human_message():
    from src.middleware.code_block_guard_middleware import CodeBlockGuardMiddleware

    middleware = CodeBlockGuardMiddleware()
    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return _response("```python\nregister(async def callback():\n    pass)\n```")

    asyncio.run(middleware.awrap_model_call(_request(), handler))

    assert len(calls) == 2
    assert isinstance(calls[1].messages[-1], HumanMessage)
