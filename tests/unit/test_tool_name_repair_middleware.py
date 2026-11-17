"""Exercise tool-name validation through model wrappers and a running graph."""

import asyncio
from uuid import uuid4

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from src.middleware.tool_name_repair_middleware import APOLOGY, ToolNameRepairMiddleware

QUERY = "langchainDocs_query_docs_filesystem_docs_by_lang_chain"
SEARCH = "langchainDocs_search_docs_by_lang_chain"


def tool(name):
    return StructuredTool.from_function(
        lambda: "documentation", name=name, description="Read docs"
    )


def call(name, content=""):
    return AIMessage(
        content=content, tool_calls=[{"name": name, "args": {}, "id": "lookup"}]
    )


class Model(FakeMessagesListChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(
            messages, stop=stop, run_manager=run_manager, **kwargs
        )
        result.generations[0].message = result.generations[0].message.model_copy(
            update={"id": str(uuid4())}
        )
        return result

    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "name,expected",
    [
        (QUERY, QUERY),
        ("langgraphDocs_query_docs_filesystem_docs_by_lang_chain", QUERY),
        ("langgraphDocs_search_docs_by_lang_chain", SEARCH),
        (
            "langgraphDocs_query_docs_filesystem_docs_by_lang_chain if false else "
            + QUERY,
            QUERY,
        ),
        ("langchainDoc_search_docs = " + SEARCH, SEARCH),
        (QUERY + "<ctrl95>", QUERY),
        ("<ctrl95>", None),
        ("langsmith/use-webhooks.mdx", None),
        ("read_flie<ctrl95>", "read_file"),
    ],
)
def test_names_are_repaired_or_dropped(asynchronous, name, expected):
    request = ModelRequest(
        model=Model(responses=[]),
        messages=[],
        tools=[tool(QUERY), tool(SEARCH), tool("read_file")],
        state={"messages": []},
    )
    middleware = ToolNameRepairMiddleware()
    original = call(name, "Looking up docs")

    def handler(request):
        return ModelResponse(result=[original])

    async def ahandler(request):
        return handler(request)

    response = (
        asyncio.run(middleware.awrap_model_call(request, ahandler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    calls = response.result[0].tool_calls
    assert [c["name"] for c in calls] == ([expected] if expected else [])
    if expected:
        assert calls[0]["id"] == "lookup"
        assert calls[0]["args"] == {}
    assert original.tool_calls[0]["name"] == name


@pytest.mark.parametrize("asynchronous", [False, True])
def test_dropped_empty_response_retries_once(asynchronous):
    request = ModelRequest(
        model=Model(responses=[]),
        messages=[],
        system_prompt="Original",
        tools=[tool(QUERY), tool(SEARCH)],
        state={"messages": []},
    )
    requests = []

    def handler(request):
        requests.append(request)
        return ModelResponse(result=[call("langsmith/use-webhooks.mdx")])

    async def ahandler(request):
        return handler(request)

    middleware = ToolNameRepairMiddleware()
    response = (
        asyncio.run(middleware.awrap_model_call(request, ahandler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    assert len(requests) == 2
    assert QUERY in requests[1].system_message.text
    assert SEARCH in requests[1].system_message.text
    assert "Original" in requests[1].system_message.text
    assert response.result[-1].text == APOLOGY
    assert not response.result[-1].tool_calls


@pytest.mark.parametrize("asynchronous", [False, True])
def test_three_invalid_calls_force_final_answer_and_reset_next_turn(asynchronous):
    graph = create_agent(
        Model(
            responses=[call("bad_search_docs")] * 3
            + [call("ignored", "Answer from context")]
        ),
        tools=[tool(SEARCH)],
        middleware=[ToolNameRepairMiddleware()],
    )
    inputs = {"messages": [HumanMessage(content="Find docs")]}
    state = asyncio.run(graph.ainvoke(inputs)) if asynchronous else graph.invoke(inputs)
    assert state["messages"][-1].text == "Answer from context"
    assert not state["messages"][-1].tool_calls
    assert state["tool_name_repairs"] == 3
    assert len([m for m in state["messages"] if isinstance(m, ToolMessage)]) == 2
    inputs = {
        **state,
        "messages": [*state["messages"], HumanMessage(content="Try again")],
    }
    state = asyncio.run(graph.ainvoke(inputs)) if asynchronous else graph.invoke(inputs)
    assert state["tool_name_repairs"] == 3
    assert len([m for m in state["messages"] if isinstance(m, ToolMessage)]) == 4


@pytest.mark.parametrize("asynchronous", [False, True])
def test_recursion_budget_returns_apology(asynchronous):
    graph = create_agent(
        Model(responses=[call(SEARCH)]),
        tools=[tool(SEARCH)],
        middleware=[ToolNameRepairMiddleware()],
    )
    inputs = {"messages": [HumanMessage(content="Find docs")]}
    config = {"recursion_limit": 12}
    state = (
        asyncio.run(graph.ainvoke(inputs, config))
        if asynchronous
        else graph.invoke(inputs, config)
    )
    assert state["messages"][-1].text == APOLOGY
    assert not state["messages"][-1].tool_calls


@pytest.mark.parametrize("asynchronous", [False, True])
def test_final_attempt_has_no_tools_even_if_model_ignores_it(asynchronous):
    request = ModelRequest(
        model=Model(responses=[]),
        messages=[],
        tools=[tool(SEARCH)],
        state={"messages": [], "tool_name_repairs": 2},
    )
    requests = []

    def handler(request):
        requests.append(request)
        return ModelResponse(result=[call("bad_search_docs")])

    async def ahandler(request):
        return handler(request)

    middleware = ToolNameRepairMiddleware()
    response = (
        asyncio.run(middleware.awrap_model_call(request, ahandler))
        if asynchronous
        else middleware.wrap_model_call(request, handler)
    )
    assert len(requests) == 2
    assert requests[1].tools == []
    assert requests[1].tool_choice is None
    assert response.result[-1].text == APOLOGY
    assert response.result[-1].tool_calls == []


def test_mixed_calls_preserve_valid_calls_and_only_use_registered_repairs():
    valid = call(SEARCH).tool_calls[0]
    message = AIMessage(
        content="",
        tool_calls=[
            valid,
            {**valid, "name": "bad_query_docs_filesystem", "id": "unavailable"},
            {**valid, "name": "langsmith/use-webhooks.mdx", "id": "unknown"},
        ],
    )
    request = ModelRequest(
        model=Model(responses=[]),
        messages=[],
        tools=[tool(SEARCH)],
        state={"messages": []},
    )
    response = ToolNameRepairMiddleware().wrap_model_call(
        request, lambda _: ModelResponse(result=[message])
    )
    assert response.result[-1].tool_calls == [valid]


@pytest.mark.parametrize(
    "schema",
    [
        {
            "name": SEARCH,
            "description": "Search docs",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "function": {
                "name": SEARCH,
                "description": "Search docs",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ],
)
def test_dictionary_tool_schemas_are_registered(schema):
    request = ModelRequest(
        model=Model(responses=[]), messages=[], tools=[schema], state={"messages": []}
    )
    response = ToolNameRepairMiddleware().wrap_model_call(
        request, lambda _: ModelResponse(result=[call("bad_search_docs")])
    )
    assert response.result[0].tool_calls[0]["name"] == SEARCH
