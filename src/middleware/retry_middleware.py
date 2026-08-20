# Retry middleware for model calls with exponential backoff
import asyncio
import json
import logging
import uuid
from typing import Any, Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

logger = logging.getLogger(__name__)

# Finish reasons that indicate a retryable failure (not an exception)
RETRYABLE_FINISH_REASONS = {
    "MALFORMED_FUNCTION_CALL",  # Gemini: invalid tool call syntax
}

BOUND_TOOL_NAMES = [
    "query_docs_filesystem_docs_by_lang_chain",
    "get_support_article_content",
    "search_support_articles",
    "search_docs_by_lang_chain",
    "fetch_langchain_pricing",
    "check_links",
    "write_todos",
    "read_file",
    "grep",
    "ls",
    "glob",
]


class MalformedResponseError(Exception):
    """Raised when model returns a malformed response after exhausting retries."""

    pass


def _decompose_tool_name(name: str, tool_names: list[str]) -> list[str] | None:
    ordered_names = sorted(set(tool_names), key=len, reverse=True)
    fragments = []
    remaining = name
    while remaining:
        match = next(
            (
                tool_name
                for tool_name in ordered_names
                if remaining.startswith(tool_name)
            ),
            None,
        )
        if match is None:
            return None
        fragments.append(match)
        remaining = remaining[len(match) :]
    return fragments if len(fragments) >= 2 else None


def _split_json_objects(arguments: str) -> list[dict[str, Any]] | None:
    objects = []
    position = 0
    while position < len(arguments):
        while position < len(arguments) and arguments[position].isspace():
            position += 1
        if position == len(arguments) or arguments[position] != "{":
            return None

        start = position
        depth = 0
        in_string = False
        escaped = False
        while position < len(arguments):
            character = arguments[position]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    try:
                        parsed = json.loads(arguments[start : position + 1])
                    except json.JSONDecodeError:
                        return None
                    if not isinstance(parsed, dict):
                        return None
                    objects.append(parsed)
                    position += 1
                    break
            position += 1
        else:
            return None
    return objects


def _split_merged_function_call(
    message: Any, tool_names: list[str]
) -> list[tuple[str, dict[str, Any]]] | None:
    """Split a concatenated function call when its boundaries are unambiguous."""
    function_call = getattr(message, "additional_kwargs", {}).get("function_call", {})
    name = function_call.get("name")
    arguments = function_call.get("arguments")
    if not isinstance(name, str) or not isinstance(arguments, str):
        return None
    if name in tool_names:
        return None

    tool_fragments = _decompose_tool_name(name, tool_names)
    if tool_fragments is None:
        return None
    argument_fragments = _split_json_objects(arguments)
    if argument_fragments is None or len(argument_fragments) != len(tool_fragments):
        return None
    return list(zip(tool_fragments, argument_fragments, strict=True))


class ModelRetryMiddleware(AgentMiddleware):
    def __init__(
        self,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        super().__init__()
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def _get_finish_reason(self, response: ModelResponse) -> str:
        """Extract finish_reason from response metadata."""
        message = self._get_response_message(response)
        metadata = getattr(message, "response_metadata", None) or {}
        return metadata.get("finish_reason", "")

    def _get_response_message(self, response: ModelResponse) -> Any:
        return response.result[-1]

    def _get_tool_names(self, request: ModelRequest) -> list[str]:
        tool_names = []
        for tool in getattr(request, "tools", []):
            if isinstance(tool, dict):
                name = tool.get("name") or tool.get("function", {}).get("name")
            else:
                name = getattr(tool, "name", None)
            if isinstance(name, str):
                tool_names.append(name)
        return tool_names or BOUND_TOOL_NAMES

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        last_exception: Exception | None = None
        last_retryable_reason: str | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await handler(request)
                finish_reason = self._get_finish_reason(response)
                message = self._get_response_message(response)
                tool_names = self._get_tool_names(request)
                merged_calls = _split_merged_function_call(message, tool_names)

                if merged_calls and len(merged_calls) > len(message.tool_calls):
                    message.tool_calls = [
                        {
                            "name": tool_name,
                            "args": args,
                            "id": f"call_{uuid.uuid4().hex}",
                            "type": "tool_call",
                        }
                        for tool_name, args in merged_calls
                    ]
                    message.additional_kwargs.pop("function_call", None)
                    logger.warning(
                        "Repaired merged function call; recovered tool fragments: %s",
                        ", ".join(tool_name for tool_name, _ in merged_calls),
                    )
                    return response

                function_call = getattr(message, "additional_kwargs", {}).get(
                    "function_call", {}
                )
                name = function_call.get("name")
                if (
                    isinstance(name, str)
                    and _decompose_tool_name(name, tool_names)
                    and attempt < self.max_retries
                ):
                    logger.warning(
                        "Ambiguous merged function call %s; retrying",
                        name,
                    )
                    delay = self.initial_delay * (self.backoff_factor**attempt)
                    last_retryable_reason = "MALFORMED_FUNCTION_CALL"
                    await asyncio.sleep(delay)
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

            except Exception as e:
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


__all__ = [
    "BOUND_TOOL_NAMES",
    "MalformedResponseError",
    "ModelRetryMiddleware",
    "_split_merged_function_call",
]
