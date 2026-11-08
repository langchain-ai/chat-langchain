"""Managed Deep Agent entrypoint for Chat LangChain."""

import base64
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from typing import Any

from langchain_fireworks import ChatFireworks
from langchain_fireworks import chat_models as fireworks_chat_models
from managed_deepagents import define_deep_agent

from src.agent.config import (
    DEFAULT_MODEL,
    GUARDRAILS_MODEL,
    model_retry_middleware,
    summarization_model,
    tool_retry_middleware,
)
from src.middleware.closed_book_middleware import CLOSED_BOOK_ENV, ClosedBookMiddleware
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from src.middleware.legal_handoff_middleware import LegalHandoffMiddleware
from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.middleware.trial_side_effects_middleware import TrialSideEffectsMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt
from src.tools.link_check_tools import check_links
from src.tools.pricing_tools import fetch_langchain_pricing
from src.tools.pylon_tools import get_support_article_content, search_support_articles
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata

GLM = "accounts/fireworks/models/glm-5p3-flash"
# The model of a run without the RL sidecar (an eval, or a teacher writing training data),
# with its request options.
MODEL = GLM
MODEL_KWARGS: dict[str, Any] = {"reasoning_effort": "high"}
# langchain-fireworks adds prompt_cache_key and an x-session-affinity header to a call made
# while this is set (the agent runtime sets it per thread).
_CACHE_AFFINITY = getattr(fireworks_chat_models, "_PROMPT_CACHE_AFFINITY", None)


@contextmanager
def _without_cache_affinity() -> Iterator[None]:
    if _CACHE_AFFINITY is None:
        yield
        return
    previous = _CACHE_AFFINITY.get()
    _CACHE_AFFINITY.set(None)
    try:
        yield
    finally:
        # Set, not reset: a stream may finish in another context than it started in.
        _CACHE_AFFINITY.set(previous)


# Text appended to the system message of the RL policy's calls, base64-encoded (it spans lines):
# for generating training data with extra instructions the trained model is then taught without.
POLICY_SYSTEM_ADDENDUM_ENV = "POLICY_SYSTEM_ADDENDUM_B64"


def _policy_system_addendum() -> str:
    encoded = os.environ.get(POLICY_SYSTEM_ADDENDUM_ENV)
    return base64.b64decode(encoded).decode("utf-8").strip() if encoded else ""


class PolicyFireworks(ChatFireworks):
    """GLM-5.3 Flash through Fireworks' RL sidecar (TITO).

    The sidecar fixes the reasoning effort and owns sampling and session affinity: it rejects
    a request that sets n, stop, reasoning_effort, prompt_cache_key or x-session-affinity, so
    none is sent.
    """

    @property
    def _default_params(self) -> dict:
        params = super()._default_params
        params.pop("n", None)
        if params.get("stop") is None:
            params.pop("stop", None)
        params.pop("reasoning_effort", None)
        return params

    def _create_message_dicts(
        self, messages: list[Any], stop: list[str] | None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        message_dicts, params = super()._create_message_dicts(messages, stop)
        for message in message_dicts:
            if message.get("role") == "assistant":
                # The sidecar recorded each turn's reasoning, even when empty; langchain-fireworks
                # drops an empty one, and a turn sent back without it reads as rewritten history.
                message.setdefault("reasoning_content", "")
        addendum = _policy_system_addendum()
        if addendum:
            for message in message_dicts:
                if message.get("role") == "system":
                    content = message.get("content")
                    if isinstance(content, list):
                        message["content"] = [
                            *content,
                            {"type": "text", "text": addendum},
                        ]
                    else:
                        message["content"] = f"{content or ''}\n\n{addendum}"
                    break
        return message_dicts, params

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        with _without_cache_affinity():
            return super()._generate(*args, **kwargs)

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:
        with _without_cache_affinity():
            return await super()._agenerate(*args, **kwargs)

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[Any]:
        with _without_cache_affinity():
            yield from super()._stream(*args, **kwargs)

    async def _astream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        with _without_cache_affinity():
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk


def main_model() -> ChatFireworks:
    """Build the docs agent's model.

    Under RL (TITO_POLICY_BASE_URL set) it is the policy being trained, reached through the
    trial's sidecar. The summarizer and the guardrail fallback stay on their usual route, so
    their calls are not part of the trained trajectory.
    """
    if base_url := os.environ.get("TITO_POLICY_BASE_URL"):
        # The sidecar gives its OpenAI base (…/v1); the Fireworks client adds /v1/chat/completions.
        return PolicyFireworks(
            model=GLM,
            base_url=base_url.removesuffix("/").removesuffix("/v1"),
            api_key=os.environ["TITO_POLICY_API_KEY"],
        )
    return ChatFireworks(model=MODEL, **MODEL_KWARGS)


# The MCP docs tools are declared in tools/mcp.py so the managed runtime
# owns client lifecycle and appends those tools during compilation.
docs_agent_tools = [
    search_support_articles,
    get_support_article_content,
    fetch_langchain_pricing,
    check_links,
]

docs_agent_middleware = [
    # Cap oversized user input (was auth.py). Trace metadata is applied via
    # define_deep_agent(metadata=...) so it lands on the LangSmith root run.
    IngressGuardsMiddleware(),
    # Legal and contract questions (DPA, SCC, BAA, amendments) get a fixed
    # handoff before the classifier or model can confirm legal terms.
    LegalHandoffMiddleware(),
    GuardrailsMiddleware(
        model=GUARDRAILS_MODEL.id,
        fallback_model=DEFAULT_MODEL.id,
        block_off_topic=True,
    ),
    CustomSummarizationMiddleware(
        model=DEFAULT_MODEL.id,
        summary_model=summarization_model,
        trigger=("tokens", 130_000),
        keep=("tokens", 30_000),
        summary_prompt=context_summary_prompt,
        trim_tokens_to_summarize=None,
    ),
    tool_retry_middleware,
    # No cross-model fallback: every assistant turn comes from the one model.
    model_retry_middleware,
]

# This branch runs only in RL and eval trials, which reach the live docs service: their
# feedback calls are not sent.
docs_agent_middleware.insert(0, TrialSideEffectsMiddleware())
if os.environ.get(CLOSED_BOOK_ENV):
    # Memorization probes only: retrieval tools answer "No results." (closed_book_middleware).
    docs_agent_middleware.insert(0, ClosedBookMiddleware())

agent = define_deep_agent(
    name="docs_agent",
    # A model object, so its reasoning effort is set; langchain-fireworks is a
    # declared dependency.
    model=main_model(),
    tools=docs_agent_tools,
    middleware=docs_agent_middleware,
    metadata=build_docs_agent_trace_metadata(),
)
