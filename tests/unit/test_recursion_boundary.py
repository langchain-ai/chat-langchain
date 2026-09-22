"""Tests for graceful graph recursion exhaustion handling."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.errors import GraphRecursionError

from src.middleware.recursion_boundary import with_recursion_boundary


def test_recursion_error_returns_last_non_empty_assistant_draft():
    def fail(_):
        raise GraphRecursionError("limit reached")

    runnable = with_recursion_boundary(RunnableLambda(fail))
    draft = AIMessage(content="Draft answer")

    result = runnable.invoke({"messages": [HumanMessage(content="Question"), draft]})

    assert result["messages"][-1].content == "Draft answer"


def test_recursion_error_returns_honest_message_without_draft():
    def fail(_):
        raise GraphRecursionError("limit reached")

    runnable = with_recursion_boundary(RunnableLambda(fail))

    result = runnable.invoke({"messages": [HumanMessage(content="Question")]})

    assert result["messages"][-1].content
