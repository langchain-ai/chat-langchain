"""Managed Deep Agent entrypoint for Chat LangChain."""

import inspect
import logging

from managed_deepagents import define_deep_agent

from src.agent.config import (
    DEFAULT_MODEL,
    GUARDRAILS_MODEL,
    answer_sanity_guard_middleware,
    citation_guard_middleware,
    docs_research_guard_middleware,
    duplicate_call_guard_middleware,
    model_fallback_middleware,
    model_retry_middleware,
    summarization_model,
    tool_retry_middleware,
)
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from src.middleware.summarization_middleware import CustomSummarizationMiddleware
from src.prompts.context_summary_prompt import context_summary_prompt
from src.tools.link_check_tools import check_links
from src.tools.pricing_tools import fetch_langchain_pricing
from src.tools.pylon_tools import get_support_article_content, search_support_articles
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata

logger = logging.getLogger(__name__)

# The MCP docs tools are declared in connectors/mcp.py so the managed runtime
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
    duplicate_call_guard_middleware,
    tool_retry_middleware,
    docs_research_guard_middleware,
    citation_guard_middleware,
    answer_sanity_guard_middleware,
    model_retry_middleware,
    model_fallback_middleware,
]


def _compiled_middleware_class_names(compiled_agent: object) -> set[str]:
    """Return middleware class names exposed by a compiled agent."""
    names: set[str] = set()
    configured_middleware = getattr(compiled_agent, "middleware", None)
    if configured_middleware is None:
        config = getattr(compiled_agent, "config", None)
        if isinstance(config, dict):
            configured_middleware = config.get("middleware")
    if configured_middleware is not None:
        names.update(type(middleware).__name__ for middleware in configured_middleware)

    nodes = getattr(compiled_agent, "nodes", {})
    names.update(
        node_name.split(".", 1)[0]
        for node_name in nodes
        if "." in node_name
    )
    model_node = nodes.get("model") if isinstance(nodes, dict) else None
    model_func = getattr(getattr(model_node, "bound", None), "func", None)
    pending = [model_func]
    seen: set[int] = set()
    while pending:
        function = pending.pop()
        if not inspect.isfunction(function) or id(function) in seen:
            continue
        seen.add(id(function))
        for cell in function.__closure__ or ():
            value = cell.cell_contents
            if inspect.ismethod(value) and value.__self__ is not None:
                names.add(type(value.__self__).__name__)
            elif inspect.isfunction(value):
                pending.append(value)
    return names


def _verify_docs_agent_middleware(
    compiled_agent: object, middleware: list[object]
) -> None:
    """Log missing configured middleware without interrupting startup."""
    try:
        available = _compiled_middleware_class_names(compiled_agent)
        missing = [
            type(item).__name__
            for item in middleware
            if type(item).__name__ not in available
        ]
        if missing:
            logger.error("Compiled docs agent is missing middleware: %s", ", ".join(missing))
    except Exception:
        logger.exception("Unable to verify compiled docs agent middleware")

agent = define_deep_agent(
    name="docs_agent",
    # Keep this literal so `mda deploy` can infer the provider package and
    # preflight GOOGLE_API_KEY.
    model="google_genai:gemini-3.5-flash-lite",
    tools=docs_agent_tools,
    middleware=docs_agent_middleware,
    # The current public app does not have cross-thread user memory. Keep MDA
    # managed memory off until identity scoping is ready.
    disable_memory=True,
    metadata=build_docs_agent_trace_metadata(),
)
_verify_docs_agent_middleware(agent, docs_agent_middleware)
