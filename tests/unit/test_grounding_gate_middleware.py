"""Tests for the search-only answer grounding gate."""

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.grounding_gate_middleware import GroundingGateMiddleware


def test_long_search_only_answer_gets_one_correction():
    middleware = GroundingGateMiddleware()
    state = {"messages": [], "grounding_discovery_tools": [], "grounding_content_tools": []}
    request = SimpleNamespace(
        tool_call={"name": "search_docs_by_lang_chain"},
        state=state,
    )

    middleware.wrap_tool_call(
        request,
        lambda _: ToolMessage(
            content='{"results": [{"path": "oss/python/langchain/agents.mdx"}]}',
            tool_call_id="call-1",
        ),
    )
    state["messages"] = [AIMessage(content="x" * 601)]

    update = middleware.after_model(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["grounding_gate_triggered"] is True
    assert isinstance(update["messages"][0], HumanMessage)
    assert "oss/python/langchain/agents.mdx" in update["messages"][0].content
    assert middleware.after_model(
        {**state, **update}, runtime=SimpleNamespace()
    ) is None


def test_content_tool_prevents_gate_for_each_source():
    for discovery_tool, content_tool in (
        ("search_docs_by_lang_chain", "query_docs_filesystem_docs_by_lang_chain"),
        ("search_support_articles", "get_support_article_content"),
    ):
        middleware = GroundingGateMiddleware()
        state = {"messages": [], "grounding_discovery_tools": [], "grounding_content_tools": []}
        for tool_name in (discovery_tool, content_tool):
            middleware.wrap_tool_call(
                SimpleNamespace(tool_call={"name": tool_name}, state=state),
                lambda _: ToolMessage(content="{}", tool_call_id="call-1"),
            )
        state["messages"] = [AIMessage(content="x" * 601)]
        result = middleware.after_model(state, runtime=SimpleNamespace())
        assert result is None


def test_short_search_only_answer_does_not_trigger():
    middleware = GroundingGateMiddleware()
    state = {
        "messages": [AIMessage(content="x" * 600)],
        "grounding_discovery_tools": ["search_support_articles"],
        "grounding_content_tools": [],
    }

    assert middleware.after_model(state, runtime=SimpleNamespace()) is None


def test_tool_call_tracking_stays_on_request_state():
    middleware = GroundingGateMiddleware()
    first_state = {"messages": []}
    second_state = {"messages": []}

    for state in (first_state, second_state):
        middleware.wrap_tool_call(
            SimpleNamespace(
                tool_call={"name": "search_support_articles"},
                state=state,
            ),
            lambda _: ToolMessage(
                content='{"articles": [{"id": "article-1"}]}',
                tool_call_id="call-1",
            ),
        )

    assert first_state["grounding_discovery_tools"] == ["search_support_articles"]
    assert second_state["grounding_discovery_tools"] == ["search_support_articles"]
    assert first_state["grounding_discovered_references"] == ["article-1"]
    assert second_state["grounding_discovered_references"] == ["article-1"]
