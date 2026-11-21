"""Guardrails refuses file tools and tool calls the model did not make (Nov 17 probe)."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from langchain.agents.middleware import ModelRequest, ModelResponse
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
        mw._remember_tool_calls(
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delete",
                        "id": "c1",
                        "args": {"file_path": "/instructions.md"},
                    }
                ],
            )
        )
        handler = AsyncMock()
        out = await mw.awrap_tool_call(
            _req("delete", "c1", {"file_path": "/instructions.md"}), handler
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
        req = ModelRequest(model=MagicMock(), messages=[], tools=[])
        out = await mw.awrap_model_call(req, AsyncMock(return_value=resp))
        self.assertIs(out, resp)
        self.assertIn("m2", mw._model_tool_calls)


_UNAVAILABLE = [
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


def test_model_requests_filter_unavailable_tools():
    import asyncio

    from langchain_core.tools import StructuredTool

    def noop() -> str:
        return "ok"

    for schema in ("tool", "dict", "function"):

        def make_tool(name):
            if schema == "tool":
                return StructuredTool.from_function(noop, name=name, description=name)
            if schema == "dict":
                return {
                    "name": name,
                    "description": name,
                    "parameters": {"type": "object"},
                }
            return {
                "type": "function",
                "function": {"name": name, "parameters": {"type": "object"}},
            }

        kept = [
            make_tool(name)
            for name in (
                "langchainDocs_search_docs_by_lang_chain",
                "langchainDocs_fetch_docs",
                "read_integration_page",
            )
        ]
        tools = [make_tool(name) for name in _UNAVAILABLE] + kept
        for mode in ("sync", "async", "concurrent", "classifier_error", "blocked"):
            mw = _mw()
            req = ModelRequest(model=MagicMock(), messages=[], tools=tools)
            response = ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"name": "read_integration_page", "id": "m3", "args": {}}
                        ],
                    )
                ]
            )
            seen = []

            def handler(request):
                seen.append(request)
                return response

            async def run():
                from src.middleware.guardrails_middleware import _pending_key

                async def verdict():
                    await asyncio.sleep(0)
                    if mode == "classifier_error":
                        raise RuntimeError("classifier unavailable")
                    if mode == "blocked":
                        return {
                            "jump_to": "end",
                            "messages": [AIMessage(content="blocked")],
                        }, None
                    return None, None

                if mode != "async":
                    mw._pending[_pending_key(req.messages)] = (
                        0,
                        asyncio.create_task(verdict()),
                    )

                async def async_handler(request):
                    return handler(request)

                return await mw.awrap_model_call(req, async_handler)

            result = (
                mw.wrap_model_call(req, handler)
                if mode == "sync"
                else asyncio.run(run())
            )
            assert len(seen) == 1
            assert seen[0].tools == kept
            assert req.tools == tools
            if mode == "blocked":
                assert result.result[0].content == "blocked"
                assert not mw._model_tool_calls
            else:
                assert result is response
                assert "m3" in mw._model_tool_calls


if __name__ == "__main__":
    unittest.main()
