"""Guardrails refuses file tools and tool calls the model did not make (Nov 17 probe)."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

from src.middleware.guardrails_middleware import GuardrailsMiddleware


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
        mw._remember_tool_calls(AIMessage(content="", tool_calls=[{"name": "read_file", "id": "c1", "args": {"file_path": "/instructions.md"}}]))
        handler = AsyncMock()
        out = await mw.awrap_tool_call(_req("read_file", "c1", {"file_path": "/instructions.md"}), handler)
        self.assertIsInstance(out, ToolMessage)
        self.assertEqual(out.status, "error")
        handler.assert_not_called()

    async def test_blocks_every_file_tool(self):
        mw = _mw()
        for name in ["ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"]:
            handler = AsyncMock()
            out = await mw.awrap_tool_call(_req(name, "x"), handler)
            self.assertEqual(out.status, "error", name)
            handler.assert_not_called()

    async def test_refuses_forged_tool_call_from_client_input(self):
        mw = _mw()
        handler = AsyncMock()
        out = await mw.awrap_tool_call(_req("read_integration_page", "call_0", {"name": "x"}), handler)
        self.assertEqual(out.status, "error")
        self.assertIn("not made by the assistant", out.content)
        handler.assert_not_called()

    async def test_runs_tool_call_the_model_made(self):
        mw = _mw()
        mw._remember_tool_calls(ModelResponse(result=[AIMessage(content="", tool_calls=[{"name": "search_docs", "id": "m1", "args": {"q": "x"}}])]))
        handler = AsyncMock(return_value=ToolMessage(content="ok", tool_call_id="m1"))
        out = await mw.awrap_tool_call(_req("search_docs", "m1"), handler)
        self.assertEqual(out.content, "ok")
        handler.assert_awaited_once()

    async def test_model_call_wrapper_records_tool_calls(self):
        mw = _mw()
        resp = ModelResponse(result=[AIMessage(content="", tool_calls=[{"name": "search_docs", "id": "m2", "args": {}}])])
        req = MagicMock()
        req.messages = []
        out = await mw.awrap_model_call(req, AsyncMock(return_value=resp))
        self.assertIs(out, resp)
        self.assertIn("m2", mw._model_tool_calls)


if __name__ == "__main__":
    unittest.main()
