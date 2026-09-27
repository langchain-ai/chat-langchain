"""Repair checkpointed state when an agent run fails."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.base import copy_checkpoint, create_checkpoint, uuid6
from langgraph.config import get_config
from langgraph.types import RunnableConfig

logger = logging.getLogger(__name__)

_FAILURE_MESSAGE = "I couldn't complete that turn. Please try again."
_REGISTERED_CHECKPOINTERS: dict[int, Any] = {}
_PATCHED_CHECKPOINTER_TYPES: set[type[Any]] = set()


def _thread_config(config: RunnableConfig) -> RunnableConfig:
    configurable = config.get("configurable", {})
    return {
        "configurable": {
            "thread_id": configurable["thread_id"],
            "checkpoint_ns": configurable.get("checkpoint_ns", ""),
        }
    }


def _repair_checkpoint(checkpointer: Any, config: RunnableConfig) -> bool:
    """Persist a terminal failure message for the latest failed turn."""
    latest_config = _thread_config(config)
    checkpoint_tuple = checkpointer.get_tuple(latest_config)
    if inspect.isawaitable(checkpoint_tuple) or checkpoint_tuple is None:
        return False

    messages = checkpoint_tuple.checkpoint.get("channel_values", {}).get("messages", [])
    last_human_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], HumanMessage)
        ),
        None,
    )
    if last_human_index is None:
        return False

    if (
        len(messages) == last_human_index + 2
        and isinstance(messages[-1], AIMessage)
        and messages[-1].content == _FAILURE_MESSAGE
    ):
        return False

    repaired_messages = [
        *messages[: last_human_index + 1],
        AIMessage(content=_FAILURE_MESSAGE),
    ]
    checkpoint = copy_checkpoint(checkpoint_tuple.checkpoint)
    checkpoint["channel_values"] = {
        **checkpoint["channel_values"],
        "messages": repaired_messages,
    }
    current_version = checkpoint["channel_versions"].get("messages")
    next_version = checkpointer.get_next_version(current_version, None)
    checkpoint["channel_versions"] = {
        **checkpoint["channel_versions"],
        "messages": next_version,
    }
    metadata = dict(checkpoint_tuple.metadata)
    metadata["source"] = "update"
    metadata["writes"] = {"messages": repaired_messages}
    repaired_checkpoint = create_checkpoint(
        checkpoint,
        channels=None,
        step=int(metadata.get("step", -1)) + 1,
        id=str(uuid6()),
    )
    checkpointer.put(
        checkpoint_tuple.config,
        repaired_checkpoint,
        metadata,
        {"messages": next_version},
    )
    return True


def _repair_after_error_write(
    checkpointer: Any, config: RunnableConfig, writes: Sequence[tuple[str, Any]]
) -> None:
    """Repair the thread after LangGraph records an unhandled task error."""
    if any(channel == "__error__" for channel, _ in writes):
        try:
            _repair_checkpoint(checkpointer, config)
        except Exception:
            logger.exception("Failed to repair checkpoint after agent error")


def _patch_checkpointer_type(checkpointer_type: type[Any]) -> None:
    """Patch checkpointer writes so repair happens before the run re-raises."""
    if checkpointer_type in _PATCHED_CHECKPOINTER_TYPES:
        return

    original_put_writes = checkpointer_type.put_writes

    def put_writes(
        self: Any,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        result = original_put_writes(config, writes, task_id, task_path)
        if id(self) in _REGISTERED_CHECKPOINTERS:
            _repair_after_error_write(self, config, writes)
        return result

    checkpointer_type.put_writes = put_writes

    original_aput_writes = getattr(checkpointer_type, "aput_writes", None)
    if original_aput_writes is not None:

        async def aput_writes(
            self: Any,
            config: RunnableConfig,
            writes: Sequence[tuple[str, Any]],
            task_id: str,
            task_path: str = "",
        ) -> None:
            result = await original_aput_writes(config, writes, task_id, task_path)
            if id(self) in _REGISTERED_CHECKPOINTERS:
                _repair_after_error_write(self, config, writes)
            return result

        checkpointer_type.aput_writes = aput_writes

    _PATCHED_CHECKPOINTER_TYPES.add(checkpointer_type)


class FailureRepairMiddleware(AgentMiddleware):
    """Install persisted repair for failures escaping the agent graph."""

    def before_agent(self, state: dict[str, Any], runtime: Any) -> None:
        """Wrap the active checkpointer before the graph starts."""
        config = get_config()
        checkpointer = config.get("configurable", {}).get("__pregel_checkpointer")
        if checkpointer is not None:
            _REGISTERED_CHECKPOINTERS[id(checkpointer)] = checkpointer
            _patch_checkpointer_type(type(checkpointer))
