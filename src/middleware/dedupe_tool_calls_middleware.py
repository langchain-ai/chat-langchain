"""Short-circuit duplicate tool calls within a single agent turn.

The docs agent has a documented hard cap of 2 search/read rounds per turn, but
the cap is advisory text — nothing actually enforces it. In practice the agent
loops by retrying the same `(tool_name, args)` signature, retrying the same
`.mdx` file with `rg` → `grep` → `head` → `cat` variations, or restating a
search with synonym/case/plural changes. Each loop burns 5–10× the intended
token budget and inflates latency.

This middleware tracks every `(tool_name, canonical_json(args))` signature
issued *since the last human message*. The first occurrence runs normally; a
second occurrence is short-circuited with a synthetic `ToolMessage` so the LLM
gets feedback ("you already called this — use the prior result") without
spending another tool invocation.

The signature set is keyed by the index of the most recent `HumanMessage` in
state — that's the "turn boundary". When a new human message arrives the index
moves forward and the cached set is discarded, so dedup only applies within a
single user turn.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


def _canonical_args(args: Any) -> str:
    """Return a stable JSON string for a tool-call args payload.

    `sort_keys=True` makes `{"a":1,"b":2}` and `{"b":2,"a":1}` compare equal.
    Non-JSON-serializable values fall back to `repr` so signatures are still
    comparable even for exotic argument types.
    """
    try:
        return json.dumps(args, sort_keys=True, default=repr, ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive
        return repr(args)


def _last_human_index(messages: list[Any]) -> int:
    """Index of the latest HumanMessage in `messages`, or -1 if none.

    Used as a turn boundary: signatures recorded before this index belong to
    earlier turns and must not block calls on the current turn.
    """
    for idx in range(len(messages) - 1, -1, -1):
        if isinstance(messages[idx], HumanMessage):
            return idx
    return -1


class DedupeToolCallsMiddleware(AgentMiddleware[AgentState]):
    """Skip a tool invocation if its signature was already issued this turn.

    Signature = `(tool_name, canonical_json_of_args)`. The set is keyed on the
    index of the latest `HumanMessage` so that new turns automatically reset
    the dedup window without any explicit teardown call.
    """

    def __init__(self) -> None:
        super().__init__()
        # Map of (run_id, last_human_index) -> set of seen signatures.
        # In practice each agent run resets state, so this dict is small, but
        # keying on `id(state)` plus turn index makes the middleware safe to
        # share across concurrent runs.
        self._seen: dict[tuple[int, int], set[tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    # Signature bookkeeping
    # ------------------------------------------------------------------
    def _signature(self, request: ToolCallRequest) -> tuple[str, str]:
        tool_call = request.tool_call
        name = tool_call.get("name", "unknown_tool")
        args = tool_call.get("args", {})
        return (name, _canonical_args(args))

    def _seen_set(self, request: ToolCallRequest) -> set[tuple[str, str]]:
        """Return the dedup set for the current turn, creating it if needed.

        The key combines `id(state)` (per-run identity) with the index of the
        latest HumanMessage (per-turn boundary). When the user sends a new
        message the index advances and a fresh set is created, so prior-turn
        signatures don't leak forward.
        """
        state = getattr(request, "state", None) or {}
        messages = state.get("messages", []) if isinstance(state, dict) else getattr(state, "messages", [])
        key = (id(state), _last_human_index(messages))
        bucket = self._seen.get(key)
        if bucket is None:
            bucket = set()
            self._seen[key] = bucket
        return bucket

    def _duplicate_message(
        self,
        request: ToolCallRequest,
        signature: tuple[str, str],
    ) -> ToolMessage:
        name, args_json = signature
        # Truncate args in the synthetic message body so we don't blow context
        # for tool calls with very large argument payloads.
        truncated = args_json if len(args_json) <= 400 else args_json[:400] + "...(truncated)"
        content = (
            f"This exact call (tool={name}, args={truncated}) was already made on this turn — "
            "use the prior tool result rather than re-calling. If you need different information, "
            "change the tool or arguments (and avoid synonym/case/plural variations or shell-flag "
            "retries on the same file)."
        )
        return ToolMessage(
            content=content,
            name=name,
            tool_call_id=request.tool_call.get("id", ""),
        )

    # ------------------------------------------------------------------
    # AgentMiddleware hooks
    # ------------------------------------------------------------------
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        signature = self._signature(request)
        seen = self._seen_set(request)
        if signature in seen:
            logger.info(
                "Short-circuiting duplicate tool call on this turn: tool=%s",
                signature[0],
            )
            return self._duplicate_message(request, signature)
        seen.add(signature)
        return await handler(request)

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        # Sync mirror of awrap_tool_call. The docs agent runs async, but
        # AgentMiddleware contracts call the sync path for sync agents/tests.
        signature = self._signature(request)
        seen = self._seen_set(request)
        if signature in seen:
            logger.info(
                "Short-circuiting duplicate tool call on this turn: tool=%s",
                signature[0],
            )
            return self._duplicate_message(request, signature)
        seen.add(signature)
        return handler(request)


__all__ = ["DedupeToolCallsMiddleware"]
