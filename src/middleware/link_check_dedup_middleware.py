"""Deduplicate successful link checks within a model turn."""

from __future__ import annotations

import contextvars
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

_SUFFIX = " (already validated on this turn - do not re-check these URLs)"
_URL_PATTERN = re.compile(r"https?://[^\s)<>]+")


@dataclass
class _CachedLink:
    valid: bool
    line: str


@dataclass
class _TurnState:
    links: dict[str, _CachedLink] = field(default_factory=dict)


_TURN_STATES: contextvars.ContextVar[dict[str, _TurnState]] = contextvars.ContextVar(
    "link_check_dedup_turn_states", default={}
)


class LinkCheckDedupMiddleware(AgentMiddleware):
    """Reuse successful check_links results within each human turn."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[Any]],
    ) -> Any:
        """Deduplicate successful check_links calls for the current turn."""
        if request.tool_call.get("name") != "check_links":
            return await handler(request)

        urls = self._urls(request)
        if not urls:
            return await handler(request)

        turn_state = self._turn_state(request)
        reusable_urls = [
            url
            for url in urls
            if turn_state.links.get(url) and turn_state.links[url].valid
        ]
        new_urls = [url for url in urls if url not in reusable_urls]

        if not new_urls:
            content = self._render(urls, turn_state)
            content += _SUFFIX
            return self._tool_message(request, content)

        response = await handler(
            request.override(
                tool_call={
                    **request.tool_call,
                    "args": {**request.tool_call.get("args", {}), "urls": new_urls},
                }
            )
        )
        self._cache_response(turn_state, response, new_urls)
        if not reusable_urls:
            return response

        return self._tool_message(request, self._render(urls, turn_state))

    def _urls(self, request: ToolCallRequest) -> list[str]:
        args = request.tool_call.get("args", {})
        urls = args.get("urls", []) if isinstance(args, dict) else []
        return list(dict.fromkeys(urls)) if isinstance(urls, list) else []

    def _turn_state(self, request: ToolCallRequest) -> _TurnState:
        messages = self._messages(request.state)
        latest_human_index = self._latest_human_index(messages)
        if latest_human_index < 0:
            turn_key = "no-human-turn"
        else:
            human = messages[latest_human_index]
            turn_key = str(
                getattr(human, "id", None) or f"{latest_human_index}:{human.content!r}"
            )
        states = _TURN_STATES.get()
        state = states.get(turn_key)
        if state is None:
            state = _TurnState()
            _TURN_STATES.set({turn_key: state})
        return state

    def _messages(self, state: Any) -> list[BaseMessage]:
        if isinstance(state, dict):
            messages = state.get("messages", [])
        else:
            messages = getattr(state, "messages", [])
        return list(messages) if isinstance(messages, list) else []

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if (
                isinstance(messages[index], HumanMessage)
                or getattr(messages[index], "type", None) == "human"
            ):
                return index
        return -1

    def _cache_response(
        self, state: _TurnState, response: Any, urls: list[str]
    ) -> None:
        content = str(getattr(response, "content", response))
        valid_lines = self._section_lines(content, "valid links:")
        invalid_lines = self._section_lines(content, "invalid links:")
        for url in urls:
            if url in valid_lines:
                state.links[url] = _CachedLink(True, valid_lines[url])
            elif url in invalid_lines:
                state.links[url] = _CachedLink(False, invalid_lines[url])

    def _section_lines(self, content: str, heading: str) -> dict[str, str]:
        lines = content.splitlines()
        try:
            start = next(
                index
                for index, line in enumerate(lines)
                if line.strip().lower() == heading
            )
        except StopIteration:
            return {}
        result: dict[str, str] = {}
        for line in lines[start + 1 :]:
            stripped = line.strip()
            if not stripped:
                if result:
                    break
                continue
            if not stripped.startswith("-"):
                break
            match = _URL_PATTERN.search(stripped)
            if match:
                result[match.group(0)] = line
        return result

    def _render(self, urls: list[str], state: _TurnState) -> str:
        cached = [state.links[url] for url in urls if url in state.links]
        valid = [entry.line for entry in cached if entry.valid]
        invalid = [entry.line for entry in cached if not entry.valid]
        lines = [f"Link Check Results: {len(valid)}/{len(cached)} valid", ""]
        if invalid:
            lines.extend(["Invalid links:", *invalid, ""])
        if valid:
            lines.extend(["Valid links:", *valid])
        return "\n".join(lines)

    def _tool_message(self, request: ToolCallRequest, content: str) -> ToolMessage:
        return ToolMessage(
            content=content,
            name="check_links",
            tool_call_id=request.tool_call.get("id", ""),
        )


__all__ = ["LinkCheckDedupMiddleware"]
