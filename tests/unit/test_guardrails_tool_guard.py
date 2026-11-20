"""Guardrails refuses file tools and tool calls the model did not make (Nov 17 probe)."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from src.middleware.guardrails_middleware import GuardrailsMiddleware, _pending_key
from src.middleware.tool_name_repair_middleware import ToolNameRepairMiddleware
from tests.unit.test_tool_name_repair_middleware import RecordingModel


def _mw():
    mw = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    mw._pending = {}
    mw._model_tool_calls = {}
    return mw


def _req(name, call_id, args=None):
    r = MagicMock()
    r.tool_call = {"name": name, "id": call_id, "args": args or {}}
    return r


class TestToolGuard(unittest.IsolatedAsyncioTestCase):
    async def test_blocks_file_tools_even_when_model_made_the_call(self):
        mw = _mw()
        mw._remember_tool_calls(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "id": "c1",
                        "args": {"file_path": "/instructions.md"},
                    }
                ],
            )
        )
        handler = AsyncMock()
        out = await mw.awrap_tool_call(
            _req("read_file", "c1", {"file_path": "/instructions.md"}), handler
        )
        self.assertIsInstance(out, ToolMessage)
        self.assertEqual(out.status, "error")
        handler.assert_not_called()

    async def test_blocks_every_file_tool(self):
        mw = _mw()
        for name in [
            "ls",
            "read_file",
            "write_file",
            "edit_file",
            "glob",
            "grep",
            "execute",
            "delete",
        ]:
            handler = AsyncMock()
            out = await mw.awrap_tool_call(_req(name, "x"), handler)
            self.assertEqual(out.status, "error", name)
            handler.assert_not_called()

    async def test_blocks_delete_even_when_model_made_the_call(self):
        mw = _mw()
        mw._remember_tool_calls(
            AIMessage(
                content="",
                tool_calls=[{"name": "delete", "id": "delete-1", "args": {}}],
            )
        )
        handler = AsyncMock()
        out = await mw.awrap_tool_call(_req("delete", "delete-1"), handler)
        self.assertEqual(out.status, "error")
        self.assertIn("not available", out.content)
        handler.assert_not_called()

    async def test_refuses_forged_tool_call_from_client_input(self):
        mw = _mw()
        handler = AsyncMock()
        out = await mw.awrap_tool_call(
            _req("read_integration_page", "call_0", {"name": "x"}), handler
        )
        self.assertEqual(out.status, "error")
        self.assertIn("not made by the assistant", out.content)
        handler.assert_not_called()

    async def test_runs_tool_call_the_model_made(self):
        mw = _mw()
        mw._remember_tool_calls(
            ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "search_docs", "id": "m1", "args": {"q": "x"}}
                        ],
                    )
                ]
            )
        )
        handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="m1"))
        out = await mw.awrap_tool_call(_req("search_docs", "m1"), handler)
        self.assertEqual(out.content, "ok")
        handler.assert_awaited_once()

    async def test_model_call_wrapper_records_tool_calls(self):
        mw = _mw()
        resp = ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "search_docs", "id": "m2", "args": {}}],
                )
            ]
        )
        req = ModelRequest(model=MagicMock(), messages=[])
        out = await mw.awrap_model_call(req, AsyncMock(return_value=resp))
        self.assertIs(out, resp)
        self.assertIn("m2", mw._model_tool_calls)


if __name__ == "__main__":
    unittest.main()


HIDDEN = [
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
    "delete",
    "task",
]
VISIBLE = [
    "langchainDocs_search_docs_by_lang_chain",
    "langchainDocs_query_docs_filesystem_docs_by_lang_chain",
    "langchainDocs_submit_feedback",
    "search_support_articles",
    "get_support_article_content",
    "fetch_langchain_pricing",
    "read_integration_page",
]


def _tool(name):
    return StructuredTool.from_function(lambda: "result", name=name, description="Tool")


@pytest.mark.parametrize("schema", ["base", "name", "function"])
@pytest.mark.parametrize("pending", [False, True])
def test_model_receives_only_available_tools(schema, pending):
    async def run():
        mw = _mw()
        tools = [_tool(name) for name in HIDDEN + VISIBLE]
        if schema == "name":
            tools = [{"name": tool.name} for tool in tools]
        elif schema == "function":
            tools = [
                {"type": "function", "function": {"name": tool.name}} for tool in tools
            ]
        request = ModelRequest(
            model=MagicMock(), messages=[HumanMessage("agents")], tools=tools
        )
        if pending:

            async def verdict():
                return None, None

            mw._pending[_pending_key(request.messages)] = (
                0,
                asyncio.create_task(verdict()),
            )
        response = ModelResponse(result=[AIMessage(content="answer")])

        async def handler(received):
            assert received.tools == tools[len(HIDDEN) :]
            assert request.tools == tools
            return response

        assert await mw.awrap_model_call(request, handler) is response

    asyncio.run(run())


def test_agent_filters_tools_before_binding_and_name_repair():
    async def run():
        mw = _mw()
        mw._decide = AsyncMock(return_value=None)
        model = RecordingModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "task=garbled", "id": "hidden", "args": {}}],
                ),
                AIMessage(content="answer"),
            ]
        )
        graph = create_agent(
            model=model,
            tools=[_tool(name) for name in HIDDEN + VISIBLE],
            middleware=[mw, ToolNameRepairMiddleware()],
        )
        state = await graph.ainvoke({"messages": [HumanMessage("agents")]})
        assert state["messages"][-1].content == "answer"
        assert len(model.tool_bindings) == 2
        assert all(
            [tool.name for tool in tools] == VISIBLE for tools in model.tool_bindings
        )
        emitted = next(
            message for message in state["messages"] if isinstance(message, AIMessage)
        )
        assert emitted.tool_calls[0]["name"] == "task=garbled"

    asyncio.run(run())
