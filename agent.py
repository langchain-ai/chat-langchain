"""Managed Deep Agent entrypoint for Chat LangChain."""

from managed_deepagents import define_deep_agent

from src.agent.config import (
    DEFAULT_MODEL,
    GUARDRAILS_MODEL,
    model_fallback_middleware,
    model_retry_middleware,
    summarization_model,
    tool_retry_middleware,
)
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from src.middleware.legal_handoff_middleware import LegalHandoffMiddleware
from src.middleware.link_validation_middleware import LinkValidationMiddleware
from src.middleware.search_result_trim_middleware import SearchResultTrimMiddleware
from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt
from src.tools.pricing_tools import fetch_langchain_pricing
from src.tools.pylon_tools import get_support_article_content, search_support_articles
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata

# The MCP docs tools are declared in tools/mcp.py so the managed runtime
# owns client lifecycle and appends those tools during compilation.
docs_agent_tools = [
    search_support_articles,
    get_support_article_content,
    fetch_langchain_pricing,
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
    LinkValidationMiddleware(),
    CustomSummarizationMiddleware(
        model=DEFAULT_MODEL.id,
        summary_model=summarization_model,
        trigger=("tokens", 130_000),
        keep=("tokens", 30_000),
        summary_prompt=context_summary_prompt,
        trim_tokens_to_summarize=None,
    ),
    tool_retry_middleware,
    # Docs search returns whole chunks; keep metadata plus short snippets
    # (pages are read with the filesystem tool anyway).
    SearchResultTrimMiddleware(),
    model_fallback_middleware,
    model_retry_middleware,
]

agent = define_deep_agent(
    name="docs_agent",
    # Keep this literal so `mda deploy` can infer the provider package and
    # preflight GOOGLE_API_KEY.
    model="anthropic:claude-haiku-4-5-20251001",  # experiment: middle model candidate
    tools=docs_agent_tools,
    middleware=docs_agent_middleware,
    metadata=build_docs_agent_trace_metadata(),
)
