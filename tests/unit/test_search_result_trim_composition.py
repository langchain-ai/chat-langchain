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


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("read_first", [False, True])
@pytest.mark.parametrize("heed_reminder", [False, True])
def test_search_answer_reads_page_or_receives_only_one_reminder(
    asynchronous, read_first, heed_reminder
):
    row = "| langchain-tavily | Official integration package |"

    def search(query: str) -> str:
        return "Title: Integrations\nContent: " + "| Header |\n" * 40 + row

    def read_page(path: str) -> str:
        return "Install the official integration with pip install langchain-tavily."

    search_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "langchainDocs_search_docs_by_lang_chain",
                "args": {"query": "langchain_tavily python package"},
                "id": "search",
            }
        ],
    )
    read_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "langchainDocs_query_docs_filesystem_docs_by_lang_chain",
                "args": {"path": "/integrations.mdx"},
                "id": "read",
            }
        ],
    )
    responses = [search_call]
    if read_first:
        responses += [read_call, AIMessage(content="Install langchain-tavily.")]
    else:
        responses += [AIMessage(content="The official package does not exist.")]
        if heed_reminder:
            responses += [read_call]
        responses += [AIMessage(content="Install langchain-tavily.")]
    model = RecordingModel(responses=responses)
    graph = create_deep_agent(
        model=model,
        tools=[
            StructuredTool.from_function(
                search,
                name="langchainDocs_search_docs_by_lang_chain",
                description="Search",
            ),
            StructuredTool.from_function(
                read_page,
                name="langchainDocs_query_docs_filesystem_docs_by_lang_chain",
                description="Read",
            ),
        ],
        middleware=[SearchResultTrimMiddleware()],
        backend=StateBackend(),
    )
    inputs = {"messages": [{"role": "user", "content": "Does langchain_tavily exist?"}]}
    state = asyncio.run(graph.ainvoke(inputs)) if asynchronous else graph.invoke(inputs)
    assert row in next(
        m.content for m in model.model_inputs[1] if isinstance(m, ToolMessage)
    )
    reminders = [m for m in state["messages"] if m.name == "docs_page_read_reminder"]
    assert len(reminders) == (0 if read_first else 1)
    assert state["messages"][-1].content == "Install langchain-tavily."
    assert all(
        m.content != "The official package does not exist." for m in state["messages"]
    )
    if read_first or heed_reminder:
        assert any(
            isinstance(m, ToolMessage) and "pip install langchain-tavily" in m.content
            for m in model.model_inputs[-1]
        )
