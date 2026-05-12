# Docs agent for LangChain customer service with docs and knowledge base tools
import logging
import os

from langchain.agents import create_agent

from src.agent.config import (
    GUARDRAILS_MODEL,
    configurable_model,
    model_fallback_middleware,
    model_retry_middleware,
)
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.tool_call_limit_middleware import ToolCallLimitMiddleware
from src.prompts.docs_agent_prompt import docs_agent_prompt
from src.tools.docs_tools import SearchDocsByLangChain
from src.tools.link_check_tools import check_links
from src.tools.pylon_tools import get_article_content, search_support_articles

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.info("Docs agent module loaded")

# Guardrails middleware ensures users only ask LangChain-related questions
guardrails_middleware = GuardrailsMiddleware(
    model=GUARDRAILS_MODEL.id,
    block_off_topic=True,
)
logger.info(f"Guardrails middleware using {GUARDRAILS_MODEL.name}")

# Hard cap on tool-call iterations so the agent cannot get stuck issuing
# progressively broader searches instead of synthesizing an answer.
# Tuned to ~16 tool messages (~8 rounds when calling docs+KB in parallel).
MAX_TOOL_CALLS = int(os.environ.get("DOCS_AGENT_MAX_TOOL_CALLS", "16"))
tool_call_limit_middleware = ToolCallLimitMiddleware(max_tool_calls=MAX_TOOL_CALLS)
logger.info(f"Tool-call limit middleware: max_tool_calls={MAX_TOOL_CALLS}")

docs_agent = create_agent(
    model=configurable_model,
    tools=[
        SearchDocsByLangChain,
        search_support_articles,
        get_article_content,
        check_links,
    ],
    system_prompt=docs_agent_prompt,
    middleware=[
        guardrails_middleware,
        tool_call_limit_middleware,
        model_retry_middleware,
        model_fallback_middleware,
    ],
)

if _revision_id := os.environ.get("LANGCHAIN_REVISION_ID"):
    docs_agent = docs_agent.with_config(
        {"metadata": {"LANGSMITH_AGENT_VERSION": _revision_id}}
    )
