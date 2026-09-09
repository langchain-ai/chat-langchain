"""Avoid repeating successful link checks during one agent turn."""

from __future__ import annotations

import contextvars
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

_URL_PATTERN = re.compile(r"https?://[^\s)<>]+")
_VALID_SECTION = "Valid links:"
_TURN_CACHE: contextvars.ContextVar[tuple[str, dict[str, str]] | None] = (
    contextvars.ContextVar("link_check_dedup_turn_cache", default=None)
)


class LinkCheckDedupMiddleware(AgentMiddleware):
    """Deduplicate successful check_links calls within the current turn."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Reuse successful check results within the current turn."""
        if request.tool_call.get("name") != "check_links":
            return await handler(request)

        urls = request.tool_call.get("args", {}).get("urls", [])
        if not isinstance(urls, list):
            return await handler(request)
        if not urls:
            return await handler(request)

        cache = self._cache(request)
        cached_urls = [url for url in urls if url in cache]
        new_urls = [url for url in urls if url not in cache]
        if not new_urls:
            return self._tool_message(request, self._format_results(urls, cache, {}))

        response = await handler(
            request.override(tool_call={**request.tool_call, "args": {**request.tool_call.get("args", {}), "urls": new_urls}})
        )
        if not isinstance(response, ToolMessage):
            return response

        fresh_valid, fresh_invalid = self._result_lines(response.content)
        cache.update(fresh_valid)
        if not cached_urls:
            return response
        return self._tool_message(
            request,
            self._format_results(urls, cache, fresh_invalid),
        )

    def _cache(self, request: ToolCallRequest) -> dict[str, str]:
        messages = self._messages(request.state)
        turn_key = self._turn_key(messages)
        current = _TURN_CACHE.get()
        if current is None or current[0] != turn_key:
            current = (turn_key, {})
            _TURN_CACHE.set(current)
        return current[1]

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", [])
        return list(messages)

    def _turn_key(self, messages: list[BaseMessage]) -> str:
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                human = messages[index]
                return str(getattr(human, "id", None) or f"{index}:{human.content!r}")
        return "no-human-turn"

    def _result_lines(self, content: Any) -> tuple[dict[str, str], dict[str, str]]:
        text = content if isinstance(content, str) else str(content)
        valid: dict[str, str] = {}
        invalid: dict[str, str] = {}
        in_valid = False
        in_invalid = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == _VALID_SECTION:
                in_valid = True
                in_invalid = False
                continue
            if stripped == "Invalid links:":
                in_invalid = True
                in_valid = False
                continue
            if not stripped.startswith("-"):
                continue
            urls = _URL_PATTERN.findall(line)
            if not urls:
                continue
            target = valid if in_valid else invalid if in_invalid else None
            if target is not None:
                url = (
                    line.strip()[2:].split(": ", 1)[0]
                    if in_invalid
                    else urls[0]
                )
                target[url] = line.strip()
        return valid, invalid

    def _format_results(
        self,
        urls: list[str],
        cached: dict[str, str],
        fresh_invalid: dict[str, str],
    ) -> str:
        valid_lines = [cached[url] for url in urls if url in cached]
        invalid_lines = [fresh_invalid[url] for url in urls if url in fresh_invalid]
        lines = [f"Link Check Results: {len(valid_lines)}/{len(urls)} valid", ""]
        if invalid_lines:
            lines.extend(["Invalid links:", *invalid_lines, ""])
        if valid_lines:
            lines.extend(["Valid links:", *valid_lines])
        lines.append("(already validated on this turn - do not re-check these URLs)")
        return "\n".join(lines)

    def _tool_message(self, request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name="check_links",
            tool_call_id=request.tool_call.get("id", ""),
        )


__all__ = ["LinkCheckDedupMiddleware"]
