import asyncio
from types import SimpleNamespace

from src.middleware.trial_side_effects_middleware import (
    STUBBED_RESULT,
    TrialSideEffectsMiddleware,
    is_stubbed,
)


def request(name: str):
    return SimpleNamespace(tool_call={"name": name, "id": "call_9", "args": {}})


def test_docs_feedback_is_stubbed_by_plain_or_prefixed_name():
    assert is_stubbed("submit_feedback")
    assert is_stubbed("langchainDocs_submit_feedback")
    assert not is_stubbed("langchainDocs_search_docs_by_lang_chain")


def test_a_stubbed_call_is_acknowledged_without_running():
    ran = []
    out = TrialSideEffectsMiddleware().wrap_tool_call(
        request("langchainDocs_submit_feedback"), lambda r: ran.append(r)
    )
    assert ran == []
    assert out.content == STUBBED_RESULT and out.tool_call_id == "call_9"


def test_other_calls_run_as_usual():
    async def handler(r):
        return "ran"

    middleware = TrialSideEffectsMiddleware()
    assert middleware.wrap_tool_call(request("check_links"), lambda r: "ran") == "ran"
    assert asyncio.run(middleware.awrap_tool_call(request("ls"), handler)) == "ran"
