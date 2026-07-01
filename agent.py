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
from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt
from src.tools.link_check_tools import check_links
from src.tools.pricing_tools import fetch_langchain_pricing
from src.tools.pylon_tools import get_support_article_content, search_support_articles

# The MCP docs tools are declared in connectors/mcp.py so the managed runtime
# owns client lifecycle and appends those tools during compilation.
docs_agent_tools = [
    search_support_articles,
    get_support_article_content,
    fetch_langchain_pricing,
    check_links,
]

docs_agent_middleware = [
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
    # Keep this literal so `mda deploy` can infer the provider package and
    # preflight GOOGLE_API_KEY.
    model="google_genai:gemini-3.1-flash-lite",
    tools=docs_agent_tools,
    middleware=docs_agent_middleware,
    # The current public app does not have cross-thread user memory. Keep MDA
    # managed memory off until identity scoping is ready.
    disable_memory=True,
)
