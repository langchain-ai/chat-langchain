"""Regression tests for ToolRetryMiddleware error-content normalization.

Background: when a dynamically-bound tool (e.g. a Gmail send tool) fails, its
wrapper can emit a ToolMessage with `status="error"` and `content="nothing"`.
The LLM reads only the content field, treats "nothing" as a successful empty
result, and confidently tells the user the email was sent. The middleware must
rewrite such messages so the failure is unambiguous in the content field.
"""

import asyncio

from langchain_core.messages import ToolMessage

from src.middleware.tool_retry_middleware import ToolRetryMiddleware


def _request(tool_name: str = "gmail_send_email", call_id: str = "call-1"):
    class _Req:
        tool_call = {"name": tool_name, "id": call_id}

    return _Req()


def _run(coro):
    return asyncio.run(coro)


def test_handler_error_status_with_nothing_content_is_rewritten():
    """A handler returning status=error, content="nothing" must be rewritten with [TOOL ERROR]."""
    middleware = ToolRetryMiddleware(max_attempts=1)

    async def handler(request):
        return ToolMessage(
            content="nothing",
            name="gmail_send_email",
            tool_call_id="call-1",
            status="error",
        )

    result = _run(middleware.awrap_tool_call(_request(), handler))

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "[TOOL ERROR]" in result.content
    assert "gmail_send_email" in result.content
    # The model must not be able to read this as a successful send.
    assert "nothing" != result.content.strip().lower()
    for phrase in ("email sent", "sent the email", "email has been sent"):
        assert phrase not in result.content.lower()


def test_handler_error_status_with_empty_content_is_rewritten():
    middleware = ToolRetryMiddleware(max_attempts=1)

    async def handler(request):
        return ToolMessage(
            content="",
            name="some_dynamic_tool",
            tool_call_id="call-2",
            status="error",
        )

    result = _run(
        middleware.awrap_tool_call(_request("some_dynamic_tool", "call-2"), handler)
    )

    assert result.status == "error"
    assert "[TOOL ERROR]" in result.content
    assert "some_dynamic_tool" in result.content


def test_handler_error_status_with_real_message_is_prefixed():
    """Existing meaningful error content is preserved but marked with [TOOL ERROR]."""
    middleware = ToolRetryMiddleware(max_attempts=1)

    async def handler(request):
        return ToolMessage(
            content="auth token expired",
            name="gmail_send_email",
            tool_call_id="call-3",
            status="error",
        )

    result = _run(middleware.awrap_tool_call(_request(call_id="call-3"), handler))

    assert result.status == "error"
    assert result.content.startswith("[TOOL ERROR]")
    assert "auth token expired" in result.content


def test_handler_success_status_is_passthrough():
    middleware = ToolRetryMiddleware(max_attempts=1)

    async def handler(request):
        return ToolMessage(
            content="Email sent to alice@example.com",
            name="gmail_send_email",
            tool_call_id="call-4",
            status="success",
        )

    result = _run(middleware.awrap_tool_call(_request(call_id="call-4"), handler))

    assert result.status == "success"
    assert result.content == "Email sent to alice@example.com"


def test_raised_exception_path_uses_error_status():
    """When the handler raises, the synthesized ToolMessage must have status=error and a [TOOL ERROR] prefix."""
    middleware = ToolRetryMiddleware(max_attempts=1)

    async def handler(request):
        raise RuntimeError("boom")

    result = _run(middleware.awrap_tool_call(_request(call_id="call-5"), handler))

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "[TOOL ERROR]" in result.content
