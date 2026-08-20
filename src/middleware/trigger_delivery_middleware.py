"""Terminal-delivery contract for trigger-sourced (scheduled) agent runs.

A cron/schedule run has no interactive reader: the final assistant message is
written into a thread nobody opens, and files the agent writes live in the
ephemeral run sandbox. Without a contract the model can satisfy a "build me a
page/report" instruction with a ``write_file`` into ``/tmp/`` and report success,
so the result is silently lost while the run looks healthy.

This middleware makes delivery explicit for those runs only: it injects the
delivery contract into the system message, and refuses to let the run terminate
until a delivery tool has actually been called (or the model has been told to
say that no channel is configured).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

#: ``runtime.identity.source.provider`` values that mean "no interactive user".
#: ``schedule`` is what MDA reports for cron runs; the others are accepted so the
#: guard still holds if a host labels the same thing differently.
TRIGGER_SOURCE_PROVIDERS = frozenset({"schedule", "trigger", "cron"})

#: Tools whose output actually leaves the run and reaches the user.
DELIVERY_TOOLS = frozenset(
    {
        "slack_write_private_message",
        "slack_send_channel_message",
        "gmail_send_email",
        "message_user",
    }
)

#: How many times a run may be pushed back to the model to deliver its result.
#: One is enough for the model to act; more risks a scheduled run looping.
MAX_DELIVERY_REPROMPTS = 1

DELIVERY_CONTRACT = """
<scheduled_run_delivery>
This run was started by a schedule, not by a person watching the conversation.
Nothing you leave behind in the thread or in the run sandbox reaches the user.

- Files you write are sandbox-only and ephemeral. Paths under `/tmp/` (and any
  other sandbox path) are not reachable by the user and do not survive the run.
  Never present a filesystem path as the deliverable or as "where the result is".
- If the request asks for a hosted, live, or shareable page and you have no
  publishing capability, say so plainly and inline the content into the delivery
  channel instead.
- Finish the run by calling a delivery tool ({delivery_tools}). If none of those
  is available to you, report that no delivery channel is configured for this
  schedule rather than claiming the task is done.
</scheduled_run_delivery>
""".strip()

UNDELIVERED_REPROMPT = """
You have not delivered anything to the user yet, and this run was started by a
schedule, so the message you just wrote will not be read by anyone.

Deliver the result now through a delivery tool ({delivery_tools}). A file written
to the sandbox (for example under `/tmp/`) is not a deliverable — inline the
content into the delivery message instead of pointing at a path.

If none of those tools is available to you, do not report the task as done:
state that this scheduled run produced a result but has no delivery channel
configured, and summarize the result in your reply.
""".strip()


class TriggerDeliveryState(AgentState):
    """Agent state plus the guard's re-prompt counter."""

    delivery_reprompts: NotRequired[int]


class TriggerDeliveryGuardMiddleware(AgentMiddleware[TriggerDeliveryState]):
    """Require trigger-sourced runs to end on a user-reachable delivery tool."""

    state_schema = TriggerDeliveryState

    def __init__(
        self,
        delivery_tools: Iterable[str] = DELIVERY_TOOLS,
        max_reprompts: int = MAX_DELIVERY_REPROMPTS,
    ):
        """Configure the delivery tool names and the re-prompt budget."""
        super().__init__()
        self.delivery_tools = frozenset(delivery_tools)
        self.max_reprompts = max_reprompts

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Any],
    ) -> Any:
        """Append the delivery contract to the system message on trigger runs."""
        return handler(self._with_delivery_contract(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[Any]],
    ) -> Any:
        """Append the delivery contract to the system message on trigger runs."""
        return await handler(self._with_delivery_contract(request))

    def _with_delivery_contract(self, request: ModelRequest) -> ModelRequest:
        """Return the request, with the delivery contract added on trigger runs."""
        if not self._is_trigger_run(request.runtime):
            return request
        return request.override(system_message=self._system_message(request))

    @hook_config(can_jump_to=["model"])
    def after_model(
        self, state: TriggerDeliveryState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Send the run back to the model when it is about to end undelivered."""
        messages = state.get("messages", [])
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        if not self._is_trigger_run(runtime):
            return None
        if self._delivered(messages):
            return None
        attempts = state.get("delivery_reprompts", 0)
        if attempts >= self.max_reprompts:
            return None
        return {
            "messages": [
                HumanMessage(content=self._format(UNDELIVERED_REPROMPT))
            ],
            "delivery_reprompts": attempts + 1,
            "jump_to": "model",
        }

    def _system_message(self, request: ModelRequest) -> SystemMessage:
        """Return the request's system message with the delivery contract added."""
        contract = self._format(DELIVERY_CONTRACT)
        existing = request.system_message
        if existing is None:
            return SystemMessage(content=contract)
        if isinstance(existing.content, str):
            return SystemMessage(content=f"{existing.content}\n\n{contract}")
        return SystemMessage(content=[*existing.content, {"type": "text", "text": contract}])

    def _format(self, template: str) -> str:
        """Fill the delivery tool names into a contract template."""
        return template.format(
            delivery_tools=", ".join(f"`{name}`" for name in sorted(self.delivery_tools))
        )

    def _delivered(self, messages: Iterable[Any]) -> bool:
        """Report whether any delivery tool was invoked during the run."""
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                name = call.get("name") if isinstance(call, Mapping) else None
                if name in self.delivery_tools:
                    return True
        return False

    def _is_trigger_run(self, runtime: Any) -> bool:
        """Report whether this run was started by a schedule rather than a user."""
        return self._run_source(runtime) in TRIGGER_SOURCE_PROVIDERS

    def _run_source(self, runtime: Any) -> str | None:
        """Resolve the run's source provider from identity, then ambient config."""
        identity = getattr(runtime, "identity", None)
        if isinstance(identity, Mapping):
            source = identity.get("source")
            if isinstance(source, Mapping):
                provider = source.get("provider")
                if isinstance(provider, str):
                    return provider
        for configurable in (self._configurable(runtime), self._ambient_configurable()):
            if not isinstance(configurable, Mapping):
                continue
            for key in ("mda_source_provider", "source"):
                value = configurable.get(key)
                if isinstance(value, str):
                    return value
        return None

    def _configurable(self, runtime: Any) -> Mapping[str, Any] | None:
        """Read ``configurable`` off the runtime, when it carries one."""
        config = getattr(runtime, "config", None)
        if isinstance(config, Mapping) and isinstance(config.get("configurable"), Mapping):
            return config["configurable"]
        configurable = getattr(runtime, "configurable", None)
        return configurable if isinstance(configurable, Mapping) else None

    def _ambient_configurable(self) -> Mapping[str, Any] | None:
        """Read ``configurable`` from the ambient LangGraph config, if any."""
        try:
            from langgraph.config import get_config

            config = get_config()
        except Exception:
            return None
        configurable = config.get("configurable") if isinstance(config, Mapping) else None
        return configurable if isinstance(configurable, Mapping) else None


__all__ = [
    "DELIVERY_TOOLS",
    "MAX_DELIVERY_REPROMPTS",
    "TRIGGER_SOURCE_PROVIDERS",
    "TriggerDeliveryGuardMiddleware",
]
