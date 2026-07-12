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
from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.middleware.tool_scope_middleware import ToolScopeMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt
from src.tools.link_check_tools import check_links
from src.tools.pricing_tools import fetch_langchain_pricing
from src.tools.pylon_tools import get_support_article_content, search_support_articles
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata

# The MCP docs tools are declared in connectors/mcp.py so the managed runtime
# owns client lifecycle and appends those tools during compilation.
docs_agent_tools = [
    search_support_articles,
    get_support_article_content,
    fetch_langchain_pricing,
    check_links,
]

docs_agent_middleware = [
    # docs_agent is a stateless documentation Q&A assistant with no user
    # filesystem. The deepagents harness always injects builtin filesystem +
    # planning tools and exposes no opt-out flag, so strip them here before the
    # model can call write_file and return a fabricated "Updated file ..." reply.
    ToolScopeMiddleware(),
    # Cap oversized user input (was auth.py). Trace metadata is applied via
    # define_deep_agent(metadata=...) so it lands on the LangSmith root run.
    IngressGuardsMiddleware(),
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
    model_retry_middleware,
    model_fallback_middleware,
]

agent = define_deep_agent(
    name="docs_agent",
    # Keep this literal so `mda deploy` can infer the provider package and
    # preflight GOOGLE_API_KEY.
    model="google_genai:gemini-3.1-flash-lite",
    tools=docs_agent_tools,
    middleware=docs_agent_middleware,
    # The current public app does not have cross-thread user memory. Keep MDA
    # managed memory off until identity scoping is ready.
    disable_memory=True,
    # Root-trace parity with legacy docs_graph.with_config(metadata=...).
    metadata=build_docs_agent_trace_metadata(),
)
