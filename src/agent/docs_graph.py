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
from src.prompts.docs_agent_prompt import docs_agent_prompt
from src.tools.link_check_tools import check_links
from src.tools.mcp_tools import mcp_docs_tools
from src.tools.pricing_tools import fetch_langchain_pricing
from src.tools.pylon_tools import get_support_article_content, search_support_articles

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.info("Docs agent module loaded")

# Guardrails middleware ensures users only ask LangChain-related questions
guardrails_middleware = GuardrailsMiddleware(
    model=GUARDRAILS_MODEL.id,
    block_off_topic=True,
)
logger.info(f"Guardrails middleware using {GUARDRAILS_MODEL.name}")

docs_agent_tools = [
    *mcp_docs_tools,
    search_support_articles,
    get_support_article_content,
    fetch_langchain_pricing,
    check_links,
]

docs_agent_middleware = [
    guardrails_middleware,
    model_retry_middleware,
    model_fallback_middleware,
]

docs_agent = create_agent(
    model=configurable_model,
    tools=docs_agent_tools,
    system_prompt=docs_agent_prompt,
    middleware=docs_agent_middleware,
)
docs_agent.tools = docs_agent_tools
docs_agent.middleware = docs_agent_middleware

if _revision_id := os.environ.get("LANGCHAIN_REVISION_ID"):
    docs_agent = docs_agent.with_config(
        {"metadata": {"LANGSMITH_AGENT_VERSION": _revision_id}}
    )
    docs_agent.tools = docs_agent_tools
    docs_agent.middleware = docs_agent_middleware
