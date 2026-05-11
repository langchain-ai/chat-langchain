# Docs agent for LangChain customer service with docs and knowledge base tools
import logging
import os

from langchain.agents import create_agent
from langsmith import Client

from src.agent.config import (
    GUARDRAILS_MODEL,
    configurable_model,
    model_fallback_middleware,
    model_retry_middleware,
)
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.guardrails_middleware import (
    guardrails_prompt_commit,
    guardrails_prompt_source,
)
from src.prompts.docs_agent_prompt import docs_agent_prompt as _local_prompt
from src.tools.link_check_tools import check_links
from src.tools.mcp_tools import mcp_docs_tools
from src.tools.pricing_tools import fetch_langchain_pricing
from src.tools.pylon_tools import get_support_article_content, search_support_articles

# Set up logging for this module
logger = logging.getLogger(__name__)
logger.info("Docs agent module loaded")

_USE_STAGING = (
    os.getenv("LANGSMITH_HOST_PROJECT_NAME") == "immanuel-chat-langchain-test"
    or os.getenv("LANGSMITH_ENV") == "dev"
)
_PROMPT_HUB_NAME = (
    "public-chat-langchain-test:staging"
    if _USE_STAGING
    else "public-chat-langchain-test:production"
)
_langsmith_client = Client()
try:
    _prompt_template = _langsmith_client.pull_prompt(_PROMPT_HUB_NAME)
    docs_agent_prompt = _prompt_template.invoke({"messages": []}).messages[0].content
    prompt_commit = (_prompt_template.metadata or {}).get("lc_hub_commit_hash")
    prompt_source = f"hub:{_PROMPT_HUB_NAME}"
    logger.info(
        f"Loaded prompt from hub: {_PROMPT_HUB_NAME} @ {(prompt_commit or '')[:8]}"
    )
except Exception:
    logger.warning(
        f"Failed to pull prompt from hub ({_PROMPT_HUB_NAME}), falling back to local file"
    )
    docs_agent_prompt = _local_prompt
    prompt_commit = None
    prompt_source = "local:src/prompts/docs_agent_prompt.py"

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

_prompt_metadata: dict[str, str] = {
    "prompt_source": prompt_source,
    "guardrails_prompt_source": guardrails_prompt_source,
}
if prompt_commit:
    _prompt_metadata["prompt_commit"] = prompt_commit
if guardrails_prompt_commit:
    _prompt_metadata["guardrails_prompt_commit"] = guardrails_prompt_commit
if _revision_id := os.environ.get("LANGCHAIN_REVISION_ID"):
    _prompt_metadata["LANGSMITH_AGENT_VERSION"] = _revision_id

docs_agent = docs_agent.with_config(metadata=_prompt_metadata)
docs_agent.tools = docs_agent_tools
docs_agent.middleware = docs_agent_middleware
