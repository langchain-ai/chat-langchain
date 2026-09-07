from types import SimpleNamespace

from src.middleware.docs_link_guard_middleware import DocsLinkGuardMiddleware


def _state(answer, *turn_messages):
    return {"messages": [SimpleNamespace(type="human", content="Question"), *turn_messages, answer]}


def test_removes_relevant_docs_without_check_links_call():
    answer = SimpleNamespace(
        type="ai",
        content="**Answer.**\n\n**Relevant docs:**\n- [Guide](https://docs.langchain.com/legacy/path)\n",
        tool_calls=[],
    )

    update = DocsLinkGuardMiddleware().after_model(_state(answer), runtime=None)

    assert update["messages"][0].content == "**Answer.**\n\n"


def test_preserves_link_returned_by_check_links():
    check_result = SimpleNamespace(
        type="tool",
        name="check_links",
        content="Link Check Results: 1/1 valid\n\nValid links:\n  - https://docs.langchain.com/guide",
    )
    answer = SimpleNamespace(
        type="ai",
        content="**Answer.**\n\n**Relevant docs:**\n- [Guide](https://docs.langchain.com/guide)\n",
        tool_calls=[],
    )

    update = DocsLinkGuardMiddleware().after_model(_state(answer, check_result), runtime=None)

    assert update is None
    assert "https://docs.langchain.com/guide" in answer.content
