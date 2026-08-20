"""Delivery guard: unattended runs must end in a user-reachable delivery.

Runs started by a schedule, cron, or trigger have nobody watching the thread, so
a file written to the ephemeral run sandbox is never seen by the user. This
middleware blocks a terminal "I created it" message on such runs when no
delivery tool was invoked, and raises a ``message_user`` interrupt naming the
missing destination instead.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime
from langgraph.types import interrupt

#: Run sources that have no human attached to the thread.
UNATTENDED_RUN_SOURCES = frozenset(
    {"cron", "schedule", "scheduled", "trigger", "triggered", "webhook"}
)

#: Config/context keys that carry the run source, in lookup order.
_RUN_SOURCE_KEYS = ("run_source", "trigger_source", "source")

#: Tools that put a result somewhere the user can actually reach it.
DELIVERY_TOOL_NAMES = frozenset({"gmail_send_email", "message_user", "send_email"})

#: Tool-name fragments that mark a delivery tool (``slack_post_message``, ...).
DELIVERY_TOOL_MARKERS = ("send", "post", "message")

#: Prefixes of tool families whose send/post members deliver to the user.
DELIVERY_TOOL_PREFIXES = ("slack_",)

#: Tools that only write into the ephemeral run sandbox.
SANDBOX_WRITE_TOOLS = frozenset({"write_file", "edit_file"})

#: Sandbox write args that hold the target path, in lookup order.
_PATH_ARG_KEYS = ("file_path", "path", "filename")

#: Phrases in a final message that assert the deliverable reached the user.
DELIVERY_CLAIM_MARKERS = (
    "created",
    "built",
    "posted",
    "published",
    "sent",
    "live page",
    "dashboard",
    "you can open",
    "you can view",
)


class DeliveryGuardMiddleware(AgentMiddleware[AgentState]):
    """Require unattended runs to deliver their result instead of claiming success."""

    def __init__(self, *, extra_delivery_tools: Sequence[str] | None = None) -> None:
        """Initialize the guard, optionally recognizing extra delivery tools."""
        super().__init__()
        self.delivery_tools = DELIVERY_TOOL_NAMES | frozenset(extra_delivery_tools or ())

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Swap an undelivered success claim for a missing-destination interrupt."""
        if not self._is_unattended(runtime):
            return None

        messages = state.get("messages", [])
        final = messages[-1] if messages else None
        if not isinstance(final, AIMessage) or final.tool_calls:
            return None

        tool_calls = list(self._tool_calls(messages))
        if any(self._is_delivery_tool(name) for name, _ in tool_calls):
            return None

        artifacts = self._sandbox_artifacts(tool_calls)
        if not artifacts and not self._claims_delivery(final):
            return None

        notice = self._missing_delivery_notice(artifacts)
        interrupt(
            {
                "type": "message_user",
                "reason": "missing_delivery_destination",
                "message": notice,
                "sandbox_artifacts": artifacts,
            }
        )
        # Same id => the messages reducer overwrites the unreachable claim in place.
        return {"messages": [AIMessage(content=notice, id=final.id)]}

    def _is_unattended(self, runtime: Runtime) -> bool:
        """Return True when the run was started by a schedule, cron, or trigger."""
        source = self._run_source(runtime)
        return source in UNATTENDED_RUN_SOURCES

    def _run_source(self, runtime: Runtime) -> str | None:
        """Read the run source from the runtime context or the run config."""
        for scope in (getattr(runtime, "context", None), *self._config_scopes()):
            value = self._lookup(scope, _RUN_SOURCE_KEYS)
            if value:
                return value.strip().lower()
        return None

    def _config_scopes(self) -> tuple[Any, ...]:
        """Return the run config sections that may carry the run source."""
        try:
            config = get_config()
        except Exception:
            return ()
        return (config.get("metadata"), config.get("configurable"))

    def _lookup(self, scope: Any, keys: Sequence[str]) -> str | None:
        """Return the first string value found for ``keys`` in a dict or object."""
        if scope is None:
            return None
        for key in keys:
            value = scope.get(key) if isinstance(scope, dict) else getattr(scope, key, None)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _tool_calls(self, messages: Sequence[Any]) -> Iterable[tuple[str, dict[str, Any]]]:
        """Yield ``(name, args)`` for every tool call made during the run."""
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                name = call.get("name") or ""
                args = call.get("args") or {}
                yield name, args if isinstance(args, dict) else {}

    def _is_delivery_tool(self, name: str) -> bool:
        """Return True when the tool sends the result to the user."""
        if name in self.delivery_tools:
            return True
        return name.startswith(DELIVERY_TOOL_PREFIXES) and any(
            marker in name for marker in DELIVERY_TOOL_MARKERS
        )

    def _sandbox_artifacts(self, tool_calls: Sequence[tuple[str, dict[str, Any]]]) -> list[str]:
        """Return the sandbox paths written during the run, in call order."""
        artifacts: list[str] = []
        for name, args in tool_calls:
            if name not in SANDBOX_WRITE_TOOLS:
                continue
            path = self._lookup(args, _PATH_ARG_KEYS)
            if path and path not in artifacts:
                artifacts.append(path)
        return artifacts

    def _claims_delivery(self, message: AIMessage) -> bool:
        """Return True when the final message asserts the deliverable reached the user."""
        lowered = str(message.text).lower()
        return any(marker in lowered for marker in DELIVERY_CLAIM_MARKERS)

    def _missing_delivery_notice(self, artifacts: Sequence[str]) -> str:
        """Explain that the unattended run has no destination for its result."""
        lines = [
            "This run was started by a schedule or trigger and has no delivery "
            "destination configured, so its result was not sent anywhere.",
        ]
        if artifacts:
            lines.append(
                "The following files exist only in the ephemeral run sandbox and "
                "cannot be opened: " + ", ".join(artifacts) + "."
            )
        lines.append(
            "Add a delivery channel to this schedule (Slack message or email) and "
            "the result will be sent there on the next run."
        )
        return " ".join(lines)


__all__ = [
    "DELIVERY_CLAIM_MARKERS",
    "DELIVERY_TOOL_NAMES",
    "SANDBOX_WRITE_TOOLS",
    "UNATTENDED_RUN_SOURCES",
    "DeliveryGuardMiddleware",
]
