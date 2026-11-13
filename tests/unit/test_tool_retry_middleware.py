"""Tests for complete single-page documentation reads."""

import re

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware

DOCS_TOOL = "langchainDocs_query_docs_filesystem_docs_by_lang_chain"
PATH = "/oss/python/page.mdx"
PAGE = "".join(f"Section {i}: complete content\n" for i in range(1, 46))


def output(text, call_id="read", status="success"):
    return ToolMessage(
        content=[{"type": "text", "text": f"exit: 0\n--- stdout ---\n{text}"}],
        tool_call_id=call_id,
        name=DOCS_TOOL,
        status=status,
    )


def filesystem(command, limit=100):
    lines = PAGE.splitlines(keepends=True)
    if command.startswith("cat"):
        text = PAGE
    elif command.startswith("head"):
        count = int(re.search(r"\d+", command)[0])
        text = "".join(lines[:count])
    elif command.startswith("tail"):
        start = int(re.search(r"\d+", command)[0])
        text = "".join(lines[start - 1 :])
    else:
        start, end = map(int, re.search(r"(\d+),(\d+)p", command).groups())
        text = "".join(lines[start - 1 : end])
    return text[:limit] + "\n... (output truncated)" if len(text) > limit else text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command,start,end",
    [
        (f"cat {PATH}", 1, 45),
        (f"sed -n '8,29p' {PATH}", 8, 29),
        (f"tail -n +12 {PATH}", 12, 45),
        (f"head -20 {PATH}", 1, 20),
        (f"head -n 200 {PATH}", 1, 45),
        (f"sed -n '8,100p' {PATH}", 8, 45),
    ],
)
async def test_recovers_requested_content_with_headers_and_partial_line(
    command, start, end
):
    request = ToolCallRequest(
        tool_call={"name": DOCS_TOOL, "args": {"command": command}, "id": "read"},
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(request):
        return output(filesystem(request.tool_call["args"]["command"]))

    result = await ToolRetryMiddleware().awrap_tool_call(request, handler)
    assert result.content == "".join(PAGE.splitlines(keepends=True)[start - 1 : end])
    assert result.tool_call_id == "read"
    assert result.name == DOCS_TOOL


@pytest.mark.asyncio
@pytest.mark.parametrize("cut", [False, True])
@pytest.mark.parametrize(
    "command,tool_name",
    [
        (f"cat {PATH}", DOCS_TOOL),
        (f"rg pattern {PATH}", DOCS_TOOL),
        (f"cat {PATH} other.mdx", DOCS_TOOL),
        (f"cat {PATH}; pwd", DOCS_TOOL),
        (f"cat {PATH}", "other_tool"),
    ],
)
async def test_preserves_non_truncated_or_unrelated_results(command, tool_name, cut):
    if cut and command == f"cat {PATH}" and tool_name == DOCS_TOOL:
        return
    message = output("text\n... (output truncated)" if cut else "complete\n")
    request = ToolCallRequest(
        tool_call={"name": tool_name, "args": {"command": command}, "id": "read"},
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(request):
        return message

    assert await ToolRetryMiddleware().awrap_tool_call(request, handler) is message


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "error", "long_line"])
async def test_failed_recovery_exposes_remaining_range(failure):
    request = ToolCallRequest(
        tool_call={
            "name": DOCS_TOOL,
            "args": {"command": f"sed -n '8,29p' {PATH}"},
            "id": "read",
        },
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(request):
        if request.tool_call["args"]["command"].startswith("sed -n '8,29p'"):
            return output("complete\npartial\n... (output truncated)")
        if failure == "exception":
            raise RuntimeError("offline")
        if failure == "error":
            return output("failed", status="error")
        return output("partial\n... (output truncated)")

    result = await ToolRetryMiddleware().awrap_tool_call(request, handler)
    assert "complete\n" in result.content
    assert f"{PATH}, unread lines 9-29" in result.content
    assert "Explicitly disclose" in result.content
    assert result.tool_call_id == "read"


class ResearchModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task",
    [
        "Summarize the whole page.",
        "Rewrite the whole page.",
        "Translate the whole page.",
        "Reorganize this documentation page concisely.",
        "请用中文重新整理这篇文档，简洁清晰地呈现内容。",
        "请翻译这篇文档。",
        "请总结这篇文档。",
        "请重新整理整篇文档，保留所有章节。",
    ],
)
@pytest.mark.parametrize("limit", [100, 10000])
@pytest.mark.parametrize(
    "command",
    [
        f"cat {PATH}",
        f"head -n 5 {PATH}",
        f"sed -n '8,12p' {PATH}",
        f"tail -n +12 {PATH}",
    ],
)
async def test_whole_page_task_receives_last_line_before_answering(
    command, limit, task
):
    async def read(command: str) -> list:
        """Read a documentation page."""
        return [
            {
                "type": "text",
                "text": f"exit: 0\n--- stdout ---\n{filesystem(command, limit=limit)}",
            }
        ]

    read.__name__ = DOCS_TOOL
    model = ResearchModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DOCS_TOOL,
                        "args": {"command": command},
                        "id": "read",
                    }
                ],
            ),
            AIMessage(content="Page summary."),
        ]
    )
    agent = create_agent(model, tools=[read], middleware=[ToolRetryMiddleware()])
    result = await agent.ainvoke({"messages": [HumanMessage(content=task)]})
    messages = result["messages"]
    tool_result = next(m for m in messages if isinstance(m, ToolMessage))
    if limit == 10000:
        assert tool_result.content == output(filesystem(command, limit)).content
    recovered = next(
        (m.text for m in messages if isinstance(m, SystemMessage)),
        tool_result.text,
    )
    assert PAGE in recovered
    assert messages[-1].content == "Page summary."


@pytest.mark.asyncio
@pytest.mark.parametrize("trailing_newline", [False, True])
@pytest.mark.parametrize("block_content", [False, True])
async def test_recovers_cut_at_line_boundary_and_preserves_eof(
    trailing_newline, block_content
):
    page = "first\n\nlast" + ("\n" if trailing_newline else "")
    request = ToolCallRequest(
        tool_call={"name": DOCS_TOOL, "args": {"command": f"cat {PATH}"}, "id": "read"},
        tool=None,
        state={},
        runtime=None,
    )

    async def handler(request):
        if request.tool_call["args"]["command"].startswith("cat"):
            text = "first\n\n... (output truncated)"
        else:
            text = page[len("first\n") :]
        result = output(text)
        if not block_content:
            result.content = result.content[0]["text"]
        return result

    result = await ToolRetryMiddleware().awrap_tool_call(request, handler)
    assert result.content == page


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_output", ["bounded", "truncated", "error"])
@pytest.mark.parametrize(
    "task",
    [
        "Rewrite this documentation page concisely.",
        "请用中文重新整理这篇文档，简洁清晰地呈现内容。",
    ],
)
async def test_whole_page_recovery_failure_is_disclosed_in_final_answer(
    initial_output, task
):
    async def read(command: str) -> list:
        """Read a documentation page."""
        if command.startswith("head"):
            if initial_output == "error":
                return [{"type": "text", "text": "exit: 1\n--- stdout ---\nfailed"}]
            if initial_output == "truncated":
                return output("first line\npartial\n... (output truncated)").content
            return output(filesystem(command, limit=10000)).content
        raise RuntimeError("offline")

    read.__name__ = DOCS_TOOL
    model = ResearchModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": DOCS_TOOL,
                        "args": {"command": f"head -n 5 {PATH}"},
                        "id": "read",
                    }
                ],
            ),
            AIMessage(content="Here is a complete rewrite."),
        ]
    )
    agent = create_agent(model, tools=[read], middleware=[ToolRetryMiddleware()])
    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=task),
            ]
        }
    )
    unread = {"error": "1-EOF", "truncated": "2-EOF", "bounded": "6-EOF"}[
        initial_output
    ]
    assert f"{PATH}, unread lines {unread}" in result["messages"][-1].content
    assert "does not cover the entire page" in result["messages"][-1].content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "task",
    [
        "Rewrite the introduction of this page.",
        "请只改写这篇文档的引言。",
        "请总结这篇文档的中间件章节。",
        "请重新整理这篇文档的第三节。",
        "请翻译这篇文档的开头。",
        "请仅精简中间件部分。",
    ],
)
async def test_subsection_request_preserves_bounded_result(task):
    message = output(filesystem(f"head -n 5 {PATH}", limit=10000))
    request = ToolCallRequest(
        tool_call={
            "name": DOCS_TOOL,
            "args": {"command": f"head -n 5 {PATH}"},
            "id": "read",
        },
        tool=None,
        state={"messages": [HumanMessage(content=task)]},
        runtime=None,
    )

    async def handler(request):
        return message

    assert await ToolRetryMiddleware().awrap_tool_call(request, handler) is message
