# Docs agent for LangChain customer service with docs and knowledge base tools
import logging

from langchain.agents import create_agent
from langchain.agents.middleware import ContextEditingMiddleware
from langchain.agents.middleware.context_editing import ClearToolUsesEdit

from src.agent.config import (
    GUARDRAILS_MODEL,
    configurable_model,
    model_fallback_middleware,
    model_retry_middleware,
)
from src.middleware.guardrails_middleware import GuardrailsMiddleware
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

# Context editing middleware clears old tool results when the conversation
# grows too large, preventing BadRequestError when history exceeds the
# 200k token limit. Trigger at 150k so there is headroom for the model's
# own response and the system prompt (~4k tokens).
context_editing_middleware = ContextEditingMiddleware(
    edits=[ClearToolUsesEdit(trigger=150_000, keep=3)],
    token_count_method="approximate",
)

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
        context_editing_middleware,
    ],
)
