"""Tests for language consistency in generated code fences."""

import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.code_language_guard_middleware import CodeLanguageGuardMiddleware


def test_python_fence_with_javascript_declaration_is_flagged():
    middleware = CodeLanguageGuardMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(), messages=[HumanMessage(content="Show me the Python example.")]
    )

    async def handler(call: ModelRequest) -> ModelResponse:
        calls.append(call)
        return ModelResponse(
            result=[AIMessage(content="```python\nconst x = 1;\n```")]
        )

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 2
    assert "explicit python language token" in calls[1].system_prompt
    assert "```python" not in result.result[0].content
    assert result.result[0].content.startswith(
        "A verified Python example could not be produced"
    )


def test_python_fence_with_snake_case_is_accepted():
    middleware = CodeLanguageGuardMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(), messages=[HumanMessage(content="Show me the Python example.")]
    )

    async def handler(call: ModelRequest) -> ModelResponse:
        calls.append(call)
        return ModelResponse(
            result=[
                AIMessage(
                    content=[
                        {"type": "text", "text": "```python\nvalue = snake_case\n```"}
                    ]
                )
            ]
        )

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
    assert result.result[0].content[0]["text"].startswith("```python")


def test_typescript_fence_with_camel_case_is_not_flagged():
    middleware = CodeLanguageGuardMiddleware()
    calls: list[ModelRequest] = []
    request = ModelRequest(
        model=object(), messages=[HumanMessage(content="Show me the TypeScript example.")]
    )

    async def handler(call: ModelRequest) -> ModelResponse:
        calls.append(call)
        return ModelResponse(
            result=[AIMessage(content="```ts\nconst todoListMiddleware = () => {};\n```")]
        )

    result = asyncio.run(middleware.awrap_model_call(request, handler))

    assert len(calls) == 1
    assert "todoListMiddleware" in result.result[0].content
