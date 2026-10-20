"""Managed Deep Agent entrypoint for Chat LangChain."""

import os

from managed_deepagents import define_deep_agent

from src.agent.config import (
    DEFAULT_MODEL,
    GUARDRAILS_MODEL,
    model_fallback_middleware,
    model_retry_middleware,
    summarization_model,
    tool_retry_middleware,
)
from src.agent.evaluation_model import get_evaluation_model
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from src.middleware.legal_handoff_middleware import LegalHandoffMiddleware
from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt
from src.tools.link_check_tools import check_links
from src.tools.pricing_tools import fetch_langchain_pricing
from src.tools.pylon_tools import get_support_article_content, search_support_articles
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata

# The MCP docs tools are declared in tools/mcp.py so the managed runtime
# owns client lifecycle and appends those tools during compilation.
docs_agent_tools = [
    search_support_articles,
    get_support_article_content,
    fetch_langchain_pricing,
    check_links,
]

evaluation_model = get_evaluation_model()

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
        collect_samples=(
            os.getenv("CLC_DISABLE_GUARDRAIL_SAMPLE_COLLECTION", "").lower()
            not in {"1", "true", "yes"}
        ),
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
    model_retry_middleware,
    # Keep evaluation responses attributable to the requested model.
    *([] if evaluation_model is not None else [model_fallback_middleware]),
]

agent = define_deep_agent(
    name="docs_agent",
    model=(
        evaluation_model
        if evaluation_model is not None
        else "google_genai:gemini-3.5-flash-lite"
    ),
    tools=docs_agent_tools,
    middleware=docs_agent_middleware,
    metadata=build_docs_agent_trace_metadata(),
)
