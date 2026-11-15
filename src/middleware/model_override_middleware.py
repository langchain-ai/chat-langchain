"""Run docs-agent model calls on a configured chat model instance.

`define_deep_agent` needs the model as a literal string (so `mda deploy` can
infer the provider package), which can't carry provider settings such as
Anthropic's reasoning effort. This middleware swaps in a configured instance
before the call. It sits before the fallback middleware, so fallbacks still
replace it when the call fails.
"""

from typing import Any, Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse


class ModelOverrideMiddleware(AgentMiddleware):
    """Replace the request's model with `model` on every call."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(request.override(model=self.model))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request.override(model=self.model))


def build_docs_agent_model() -> Any:
    """claude-haiku-5-5 with adaptive thinking at low effort.

    Default effort: ~66-71/99 on regression v2 but p50 ~20s; low effort: ~63-66/99,
    p50 ~16s; thinking disabled: ~40-44/99 and replies in English to CJK/Arabic
    questions. Budget-based thinking isn't supported on this model.
    """
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(model="claude-haiku-5-5", effort="low", max_tokens=16000)
