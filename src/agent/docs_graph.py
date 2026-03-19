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
from src.middleware.message_normalization import MessageNormalizationMiddleware
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

# Message normalization middleware converts non-consecutive SystemMessages to
# HumanMessages so that Anthropic models do not raise ValueError on multi-turn
# conversations where the frontend injects a per-turn context SystemMessage.
message_normalization_middleware = MessageNormalizationMiddleware()
logger.info(f"Guardrails middleware using {GUARDRAILS_MODEL.name}")

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
        model_retry_middleware,
        model_fallback_middleware,
        message_normalization_middleware,  # innermost: runs for every model call
    ],
)

if _revision_id := os.environ.get("LANGCHAIN_REVISION_ID"):
    docs_agent = docs_agent.with_config(
        {"metadata": {"LANGSMITH_AGENT_VERSION": _revision_id}}
    )
