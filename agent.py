"""Managed Deep Agent entrypoint for Chat LangChain."""

import os

from langchain_fireworks import ChatFireworks
from managed_deepagents import define_deep_agent

from src.agent.config import (
    DEFAULT_MODEL,
    GUARDRAILS_MODEL,
    model_retry_middleware,
    summarization_model,
    tool_retry_middleware,
)
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from src.middleware.legal_handoff_middleware import LegalHandoffMiddleware
from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt
from src.tools.link_check_tools import check_links
from src.tools.pricing_tools import fetch_langchain_pricing
from src.tools.pylon_tools import get_support_article_content, search_support_articles
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata

GLM = "accounts/fireworks/models/glm-5p3-flash"


class PolicyFireworks(ChatFireworks):
    """GLM-5.3 Flash through Fireworks' RL sidecar (TITO).

    The sidecar fixes the reasoning effort and owns sampling, and rejects a request that sets
    n, stop or reasoning_effort, so none is sent.
    """

    @property
    def _default_params(self) -> dict:
        params = super()._default_params
        params.pop("n", None)
        if params.get("stop") is None:
            params.pop("stop", None)
        params.pop("reasoning_effort", None)
        return params


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
    return ChatFireworks(model=GLM, reasoning_effort="low")


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

agent = define_deep_agent(
    name="docs_agent",
    # A model object, so its reasoning effort is set; langchain-fireworks is a
    # declared dependency.
    model=main_model(),
    tools=docs_agent_tools,
    middleware=docs_agent_middleware,
    metadata=build_docs_agent_trace_metadata(),
)
