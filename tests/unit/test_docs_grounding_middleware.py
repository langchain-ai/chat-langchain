from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from src.middleware.docs_grounding_middleware import DocsGroundingMiddleware


def _runtime():
    return Runtime(context=None)


def _state(*messages, **extra):
    return {"messages": list(messages), **extra}


def test_ungrounded_product_answer_jumps_back_to_model():
    middleware = DocsGroundingMiddleware()

    result = middleware.after_model(
        _state(
            HumanMessage(content="How do I configure an agent?"),
            AIMessage(content="Use the AgentConfig class with retry_mode='safe'."),
        ),
        _runtime(),
    )

    assert result == {"grounding_guard_attempts": 1, "jump_to": "model"}


def test_retrieved_answer_with_validated_urls_can_end_turn():
    middleware = DocsGroundingMiddleware()
    url = "https://docs.langchain.com/oss/python/langchain/overview"

    result = middleware.after_model(
        _state(
            HumanMessage(content="How do agents work?"),
            ToolMessage(
                name="query_docs_filesystem_docs_by_lang_chain",
                content="Agents combine a model with tools.",
                tool_call_id="docs-1",
            ),
            ToolMessage(
                name="check_links",
                content=f"Link Check Results: 1/1 valid\n\nValid links:\n  - {url}",
                tool_call_id="links-1",
            ),
            AIMessage(content=f"Agents combine a model with tools. See {url}"),
        ),
        _runtime(),
    )

    assert result == {"grounding_guard_attempts": 0}


def test_pricing_answer_requires_pricing_tool():
    middleware = DocsGroundingMiddleware()

    result = middleware.after_model(
        _state(
            HumanMessage(content="How much does LangSmith cost?"),
            ToolMessage(
                name="query_docs_filesystem_docs_by_lang_chain",
                content="Some documentation content.",
                tool_call_id="docs-1",
            ),
            AIMessage(content="The developer plan costs $X per month."),
        ),
        _runtime(),
    )

    assert result == {"grounding_guard_attempts": 1, "jump_to": "model"}


def test_permitted_greeting_does_not_require_tools():
    middleware = DocsGroundingMiddleware()

    result = middleware.after_model(
        _state(
            HumanMessage(content="Hello!"),
            AIMessage(content="Hello! How can I help with LangChain today?"),
        ),
        _runtime(),
    )

    assert result is None
