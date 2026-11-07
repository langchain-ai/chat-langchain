import asyncio
from types import SimpleNamespace

from src.middleware.closed_book_middleware import (
    CLOSED_BOOK_RESULT,
    ClosedBookMiddleware,
    is_closed,
)


def request(name: str):
    return SimpleNamespace(tool_call={"name": name, "id": "call_1", "args": {}})


def test_retrieval_tools_are_withheld_by_plain_or_prefixed_name():
    assert is_closed("search_support_articles")
    assert is_closed("langchainDocs_search_docs_by_lang_chain")
    assert is_closed("langchainDocs_query_docs_filesystem_docs_by_lang_chain")
    assert not is_closed("write_file")
    assert not is_closed("task")


def test_a_withheld_call_never_runs_and_gets_no_information():
    ran = []
    out = ClosedBookMiddleware().wrap_tool_call(
        request("langchainDocs_search_docs_by_lang_chain"), lambda r: ran.append(r)
    )
    assert ran == []
    assert out.content == CLOSED_BOOK_RESULT and out.tool_call_id == "call_1"


def test_other_calls_run_as_usual_sync_and_async():
    assert ClosedBookMiddleware().wrap_tool_call(request("ls"), lambda r: "ran") == "ran"

    async def handler(r):
        return "ran async"

    assert asyncio.run(ClosedBookMiddleware().awrap_tool_call(request("ls"), handler)) == "ran async"
