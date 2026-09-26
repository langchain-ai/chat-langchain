"""Retry middleware for model calls with exponential backoff."""

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Iterable
from typing import Any
from uuid import uuid4

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain.agents.middleware.model_fallback import (
    _sanitize_request_for_fallback,
    _supports_anthropic_cache_control,
)
from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphBubbleUp

logger = logging.getLogger(__name__)

VALID_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


def _tool_call_name(tool_call: object) -> object:
    if isinstance(tool_call, dict):
        return tool_call.get("name")
    return getattr(tool_call, "name", None)


def _tool_call_args(tool_call: object) -> object:
    if isinstance(tool_call, dict):
        return tool_call.get("args")
    return getattr(tool_call, "args", None)


def _tool_call_id(tool_call: object) -> object:
    if isinstance(tool_call, dict):
        return tool_call.get("id")
    return getattr(tool_call, "id", None)


def _split_tool_call(
    tool_call: object,
    valid_tool_names: set[str],
) -> list[dict[str, object]] | None:
    name = _tool_call_name(tool_call)
    arguments = _tool_call_args(tool_call)
    if not isinstance(name, str) or not isinstance(arguments, str):
        return None

    name_pairs = [
        (first, second)
        for first in valid_tool_names
        for second in valid_tool_names
        if first + second == name
    ]
    if len(name_pairs) != 1:
        return None

    decoder = json.JSONDecoder()
    try:
        first_args, first_end = decoder.raw_decode(arguments)
        second_args, second_end = decoder.raw_decode(arguments, first_end)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(first_args, dict) or not isinstance(second_args, dict):
        return None
    if arguments[second_end:].strip():
        return None

    first_name, second_name = name_pairs[0]
    return [
        {"name": first_name, "args": first_args, "id": uuid4().hex},
        {"name": second_name, "args": second_args, "id": uuid4().hex},
    ]


def sanitize_tool_calls(
    messages: list[object], valid_tool_names: Iterable[str]
) -> list[object]:
    """Remove invalid tool calls and their unmatched tool messages."""
    valid_names = {name for name in valid_tool_names if isinstance(name, str)}
    paired_ids = {
        message.tool_call_id
        for message in messages
        if isinstance(message, ToolMessage)
    }
    removed_ids: set[str] = set()
    sanitized_messages: list[object] = []

    for message in messages:
        if not isinstance(message, AIMessage):
            sanitized_messages.append(message)
            continue

        sanitized_calls: list[dict[str, object]] = []
        changed = False
        for tool_call in message.tool_calls:
            name = _tool_call_name(tool_call)
            call_id = _tool_call_id(tool_call)
            split_calls = _split_tool_call(tool_call, valid_names)
            if split_calls is not None:
                changed = True
                if call_id not in paired_ids:
                    sanitized_calls.extend(split_calls)
                elif isinstance(call_id, str):
                    removed_ids.add(call_id)
                logger.warning("Splitting concatenated tool call name %r before fallback", name)
            elif isinstance(name, str) and VALID_TOOL_NAME.fullmatch(name):
                sanitized_calls.append(dict(tool_call))
            else:
                changed = True
                if isinstance(call_id, str):
                    removed_ids.add(call_id)
                logger.warning("Removing invalid tool call name %r before fallback", name)

        if changed:
            sanitized_messages.append(
                message.model_copy(update={"tool_calls": sanitized_calls})
            )
        else:
            sanitized_messages.append(message)

    if not removed_ids:
        return sanitized_messages
    return [
        message
        for message in sanitized_messages
        if not isinstance(message, ToolMessage)
        or message.tool_call_id not in removed_ids
    ]


class SanitizingModelFallbackMiddleware(ModelFallbackMiddleware):
    """Sanitize tool calls before each fallback model attempt."""

    def _fallback_request(self, request: Any, model: Any) -> Any:
        tool_names = {
            name
            for tool in request.tools
            if isinstance(
                name := tool.name if hasattr(tool, "name") else tool.get("name"),
                str,
            )
        }
        messages = sanitize_tool_calls(request.messages, tool_names)
        fallback_request = request.override(messages=messages).override(model=model)
        if _supports_anthropic_cache_control(model):
            return fallback_request
        return _sanitize_request_for_fallback(fallback_request)

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        """Try fallback models with sanitized conversation history."""
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            last_exception = error

        for fallback_model in self.models:
            try:
                return handler(self._fallback_request(request, fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as error:
                last_exception = error

        raise last_exception

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        """Try fallback models asynchronously with sanitized history."""
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except Exception as error:
            last_exception = error

        for fallback_model in self.models:
            try:
                return await handler(self._fallback_request(request, fallback_model))
            except GraphBubbleUp:
                raise
            except Exception as error:
                last_exception = error

        raise last_exception


class ModelRetryMiddleware(AgentMiddleware):
    """Retry model calls that fail transiently or return malformed output."""

    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        """Configure retry delays and attempt count."""
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def _get_finish_reason(self, response: ModelResponse) -> str:
        """Extract finish_reason from response metadata."""
        metadata = getattr(response, "response_metadata", None) or {}
        return metadata.get("finish_reason", "")

    def _get_invalid_tool_call_names(
        self, response: ModelResponse, valid_tool_names: Iterable[str]
    ) -> list[object]:
        valid_names = {name for name in valid_tool_names if isinstance(name, str)}
        names = []
        for message in response.result:
            if not isinstance(message, AIMessage):
                continue
            names.extend(
                name
                for tool_call in message.tool_calls
                if (
                    (name := _tool_call_name(tool_call)) is None
                    or not isinstance(name, str)
                    or VALID_TOOL_NAME.fullmatch(name) is None
                    or _split_tool_call(tool_call, valid_names) is not None
                )
            )
        return names

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Retry a model call when its response or failure is retryable."""
        last_exception: Exception | None = None
        last_retryable_reason: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await handler(request)
                finish_reason = self._get_finish_reason(response)
                invalid_tool_call_names = self._get_invalid_tool_call_names(
                    response,
                    (
                        tool.name if hasattr(tool, "name") else tool.get("name")
                        for tool in request.tools
                    ),
                )

                if invalid_tool_call_names:
                    reason = f"invalid tool call name(s): {invalid_tool_call_names!r}"
                    logger.warning("Model returned %s", reason)
                    if attempt < self.max_retries:
                        delay = self.initial_delay * (self.backoff_factor**attempt)
                        last_retryable_reason = reason
                        await asyncio.sleep(delay)
                        continue
                    last_retryable_reason = reason
                    continue

                if finish_reason in RETRYABLE_FINISH_REASONS:
                    if attempt < self.max_retries:
                        delay = self.initial_delay * (self.backoff_factor**attempt)
                        logger.warning(
                            f"Retryable response ({finish_reason}) "
                            f"attempt {attempt + 1}/{self.max_retries + 1}, "
                            f"retrying in {delay:.2f}s"
                        )
                        last_retryable_reason = finish_reason
                        await asyncio.sleep(delay)
                        continue

                return response

            except (ValueError, TypeError):
                raise
            except Exception as e:
                if _is_request_shape_error(e):
                    raise
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.initial_delay * (self.backoff_factor**attempt)
                    logger.warning(
                        f"Model call failed attempt {attempt + 1}/{self.max_retries + 1}: {e}, "
                        f"retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Model call failed after {self.max_retries + 1} attempts: {e}"
                    )

        # Exhausted retries - raise for fallback middleware
        if last_exception:
            raise last_exception

        if last_retryable_reason:
            raise MalformedResponseError(
                f"Model returned {last_retryable_reason} after {self.max_retries + 1} attempts"
            )

        raise RuntimeError("Unexpected state in retry middleware")


def _is_request_shape_error(error: Exception) -> bool:
    if getattr(error, "status_code", None) != 400:
        response = getattr(error, "response", None)
        if getattr(response, "status_code", None) != 400:
            return False
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        error_body = body.get("error", body)
        if isinstance(error_body, dict) and (
            error_body.get("code") == "invalid_request_error"
            or error_body.get("type") == "invalid_request_error"
        ):
            return True
    return getattr(error, "code", None) == "invalid_request_error"


__all__ = [
    "MalformedResponseError",
    "ModelRetryMiddleware",
    "SanitizingModelFallbackMiddleware",
    "VALID_TOOL_NAME",
    "sanitize_tool_calls",
]
