"""Test search trimming in the Deep Agents filesystem middleware stack."""

import ast
import asyncio
from pathlib import Path

import pytest
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import Field

from src.middleware.search_result_trim_middleware import SearchResultTrimMiddleware


class RecordingModel(FakeMessagesListChatModel):
    bound_tool_names: list[str] = Field(default_factory=list)
    model_inputs: list[list] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.model_inputs.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def test_trimming_is_wired_inside_tool_retry_wrapper():
    tree = ast.parse(Path("agent.py").read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "docs_agent_middleware"
            for target in node.targets
        )
    )
    names = [
        node.func.id if isinstance(node, ast.Call) else node.id
        for node in assignment.value.elts
    ]
    assert names.count("SearchResultTrimMiddleware") == 1
    assert names.index("SearchResultTrimMiddleware") > names.index(
        "tool_retry_middleware"
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "tool_name, malformed",
    [
        ("search_docs_by_lang_chain", False),
        ("search_docs_by_lang_chain", True),
        ("langchainDocs_search_docs_by_lang_chain", False),
        ("unrelated_tool", False),
    ],
)
def test_large_tool_results_reach_model_after_filesystem_processing(
    asynchronous, tool_name, malformed
):
    payload = "\n\n".join(
        f"Title: Result {index}\nLink: https://docs.langchain.com/{index}\n"
        f"Page: /{index}\nContent: {'documentation ' * 2000}"
        for index in range(10)
    )
    if malformed:
        payload = payload.replace("Content:", "Content without colon:")
    trimmed_search = tool_name.endswith("search_docs_by_lang_chain") and not malformed

    def search(query: str) -> str:
        return payload

    responses = [
        AIMessage(
            content="",
            tool_calls=[{"name": tool_name, "args": {"query": "docs"}, "id": "call-1"}],
        )
    ]
    if not trimmed_search:
        responses.append(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "/large_tool_results/call-1"},
                        "id": "read-1",
                    }
                ],
            )
        )
    responses.append(AIMessage(content="Done"))
    model = RecordingModel(responses=responses)
    graph = create_deep_agent(
        model=model,
        tools=[
            StructuredTool.from_function(search, name=tool_name, description="Search")
        ],
        middleware=[SearchResultTrimMiddleware()],
        backend=StateBackend(),
    )
    inputs = {"messages": [{"role": "user", "content": "Find docs"}]}
    state = asyncio.run(graph.ainvoke(inputs)) if asynchronous else graph.invoke(inputs)
    tool_messages = [
        message for message in state["messages"] if isinstance(message, ToolMessage)
    ]
    assert "read_file" in model.bound_tool_names
    if trimmed_search:
        result = tool_messages[0].content
        assert len(result) < 5000
        for index in range(10):
            assert f"Title: Result {index}\n" in result
            assert f"Link: https://docs.langchain.com/{index}\n" in result
            assert f"Page: /{index}\n" in result
        assert result.count("...") == 10
        assert "/large_tool_results/" not in result
        assert not state.get("files")
        assert tool_messages[0] in model.model_inputs[1]
    else:
        assert "/large_tool_results/call-1" in tool_messages[0].content
        assert "read_file" in tool_messages[0].content
        assert "/large_tool_results/call-1" in state["files"]
        assert "Title: Result 0" in tool_messages[1].content
        assert "No such file" not in tool_messages[1].content


@pytest.mark.parametrize("concurrent", [False, True])
def test_guardrails_omit_harness_tools_from_model_binding(concurrent):
    from unittest.mock import AsyncMock

    from src.middleware.guardrails_middleware import GuardrailsMiddleware
    from src.middleware.tool_name_repair_middleware import ToolNameRepairMiddleware

    def docs() -> str:
        return "Documentation"

    names = ["langchainDocs_search_docs_by_lang_chain", "read_integration_page"]
    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "langgraphDocs_search_docs_by_lang_chain",
                        "args": {},
                        "id": "docs-1",
                    }
                ],
            ),
            AIMessage(content="Done"),
        ]
    )
    guardrails = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    guardrails.concurrent = concurrent
    guardrails._pending = {}
    guardrails._decide = AsyncMock(return_value=None)
    graph = create_deep_agent(
        model=model,
        tools=[
            StructuredTool.from_function(docs, name=name, description=name)
            for name in names
        ],
        middleware=[guardrails, ToolNameRepairMiddleware()],
        backend=StateBackend(),
    )
    inputs = {"messages": [{"role": "user", "content": "Find docs"}]}
    state = asyncio.run(graph.ainvoke(inputs))
    assert state["messages"][-1].content == "Done"
    result = next(
        message for message in state["messages"] if isinstance(message, ToolMessage)
    )
    assert result.content == "Documentation"
    assert result.name == names[0]
    assert set(names) <= set(model.bound_tool_names)
    assert not set(model.bound_tool_names) & {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "execute",
        "delete",
        "task",
    }


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "tool_name",
    [
        "langchainDocs_search_docs_by_lang_chain",
        "langchainDocs_query_docs_filesystem_docs_by_lang_chain",
    ],
)
def test_documentation_block_ids_do_not_reach_model(tool_name, asynchronous):
    block_id = "lc_4d671a0f-bc1a-4ee3-adce-299cb887cf17"
    text = "[State](https://docs.langchain.com/oss/python/langgraph/graph-api#state)"

    def docs() -> ToolMessage:
        return ToolMessage(
            content=[{"type": "text", "text": text, "id": block_id}],
            tool_call_id="call-1",
        )

    model = RecordingModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": tool_name, "args": {}, "id": "call-1"}],
            ),
            AIMessage(content="Done"),
        ]
    )
    graph = create_deep_agent(
        model=model,
        tools=[StructuredTool.from_function(docs, name=tool_name, description="Docs")],
        middleware=[SearchResultTrimMiddleware()],
        backend=StateBackend(),
    )
    inputs = {"messages": [{"role": "user", "content": "Explain state"}]}
    if asynchronous:
        asyncio.run(graph.ainvoke(inputs))
    else:
        graph.invoke(inputs)
    result = next(
        message for message in model.model_inputs[1] if isinstance(message, ToolMessage)
    )
    assert result.content == [{"type": "text", "text": text}]
    assert block_id not in result.model_dump_json()
