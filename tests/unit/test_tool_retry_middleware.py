"""Test per-tool timeout limits and transient error recovery."""

import asyncio
import json
import shlex
import shutil
import subprocess

import pytest
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.middleware.tool_retry_middleware import ToolRetryMiddleware

FILESYSTEM = "langchainDocs_query_docs_filesystem_docs_by_lang_chain"
SEARCH = "langchainDocs_search_docs_by_lang_chain"


def request_for(name, args):
    return ToolCallRequest(
        tool_call={"name": name, "args": args, "id": "docs-call", "type": "tool_call"},
        tool=None,
        state={},
        runtime=None,
    )


@pytest.mark.parametrize("error", ["Command timed out", "Request TIMEOUT"])
@pytest.mark.parametrize("max_attempts", [1, 3, 5])
def test_filesystem_timeout_returns_guidance_with_capped_attempts(error, max_attempts):
    command = "rg documentation " + "folder/" * 30
    request = request_for(FILESYSTEM, {"command": command})
    calls = []

    async def handler(request):
        calls.append(request)
        raise RuntimeError(error)

    result = asyncio.run(
        ToolRetryMiddleware(max_attempts=max_attempts, initial_delay=0).awrap_tool_call(
            request, handler
        )
    )

    assert len(calls) == min(2, max_attempts)
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "docs-call"
    assert result.name == FILESYSTEM
    payload = json.loads(result.content)
    assert "10-second limit" in payload["message"]
    assert payload["command"] == command[:120]
    assert "rg" in payload["suggestion"]
    assert SEARCH in payload["suggestion"]


def test_search_timeout_recovers_on_retry():
    request = request_for(SEARCH, {"query": "LangGraph persistence"})
    answer = ToolMessage(content="Found documentation", tool_call_id="docs-call")
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise RuntimeError("Request timed out")
        return answer

    result = asyncio.run(
        ToolRetryMiddleware(initial_delay=0).awrap_tool_call(request, handler)
    )

    assert result is answer
    assert len(calls) == 2


@pytest.mark.parametrize("tool_name", [FILESYSTEM, SEARCH])
@pytest.mark.parametrize("max_attempts", [3, 5])
def test_non_timeout_service_errors_use_full_retry_budget(tool_name, max_attempts):
    request = request_for(tool_name, {"command": "rg docs", "query": "docs"})
    calls = []

    async def handler(request):
        calls.append(request)
        raise RuntimeError("HTTP 503 Service unavailable")

    result = asyncio.run(
        ToolRetryMiddleware(max_attempts=max_attempts, initial_delay=0).awrap_tool_call(
            request, handler
        )
    )

    assert len(calls) == max_attempts
    payload = json.loads(result.content)
    assert payload["error"] == "Tool unavailable"
    assert str(max_attempts) in payload["message"]


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("rg -rn copy /docs", "rg -n copy /docs"),
        ("rg -ril copy /docs", "rg -il copy /docs"),
        ("rg -irn copy /docs", "rg -in copy /docs"),
        ("rg -rl copy /docs", "rg -l copy /docs"),
        (
            'rg -rn "copy|credits" /docs | head -20',
            'rg -n "copy|credits" /docs | head -20',
        ),
        (
            "rg -rn 'copy;credits' /docs && rg -rl copy /docs; rg -ril copy /docs || sed -n '1p' /docs",
            "rg -n 'copy;credits' /docs && rg -l copy /docs; rg -il copy /docs || sed -n '1p' /docs",
        ),
        (r"rg -rn copy\|credits /docs", r"rg -n copy\|credits /docs"),
        (
            "rg -C 3 -e '-rn' -g '*.mdx' -rn /docs",
            "rg -C 3 -e '-rn' -g '*.mdx' -n /docs",
        ),
        ("rg -n -C 3 copy /docs", "rg -n -C 3 copy /docs"),
        ("sed -rn '1p' /docs", "sed -rn '1p' /docs"),
        ("rg -r replacement copy /docs", "rg -r replacement copy /docs"),
        ("rg --replace replacement copy /docs", "rg --replace replacement copy /docs"),
        ("rg -rn --replace=value copy /docs", "rg -rn --replace=value copy /docs"),
        ("rg -rvalue copy /docs", "rg -rvalue copy /docs"),
        ("rg -rn -- '-rl' /docs", "rg -n -- '-rl' /docs"),
        ("rg -- -rn /docs", "rg -- -rn /docs"),
        ("rg -rnX copy /docs", "rg -rnX copy /docs"),
        ("rg --unknown -rn copy /docs", "rg --unknown -rn copy /docs"),
        ("rg -rn 'unterminated", "rg -rn 'unterminated"),
        ("echo $(rg -rn copy /docs)", "echo $(rg -rn copy /docs)"),
        ("rg -rn copy /docs > output", "rg -rn copy /docs > output"),
        ("echo 'rg -rn copy /docs'", "echo 'rg -rn copy /docs'"),
        ("rg copy -rn /docs", "rg copy -rn /docs"),
    ],
)
@pytest.mark.parametrize("tool_name", [FILESYSTEM, SEARCH])
def test_docs_command_normalization(command, expected, tool_name):
    request = request_for(tool_name, {"command": command, "other": "preserved"})

    async def handler(received):
        assert received.tool_call["args"] == {
            "command": expected if tool_name == FILESYSTEM else command,
            "other": "preserved",
        }
        assert received.tool_call["id"] == "docs-call"
        return ToolMessage(content="original matches", tool_call_id="docs-call")

    result = asyncio.run(ToolRetryMiddleware().awrap_tool_call(request, handler))

    assert result.content == "original matches"
    assert request.tool_call["args"]["command"] == command


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep is not installed")
def test_normalized_filesystem_search_returns_original_matches(tmp_path):
    page = tmp_path / "models.mdx"
    page.write_text("Copy a model with model.model_copy(update={}).\n")
    request = request_for(
        FILESYSTEM, {"command": f"rg -rn model_copy {shlex.quote(str(page))}"}
    )

    async def handler(received):
        output = subprocess.run(
            shlex.split(received.tool_call["args"]["command"]),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return ToolMessage(content=output, tool_call_id="docs-call")

    result = asyncio.run(ToolRetryMiddleware().awrap_tool_call(request, handler))

    assert result.content == "1:Copy a model with model.model_copy(update={}).\n"


@pytest.mark.parametrize("tool_name", [FILESYSTEM, SEARCH])
def test_stalled_docs_call_is_rescued_by_hedged_duplicate(tool_name):
    request = request_for(tool_name, {"command": "head -50 /a.mdx", "query": "a"})
    answer = ToolMessage(content="docs", tool_call_id="docs-call")
    calls, cancelled = [], []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise
        return answer

    async def run():
        loop = asyncio.get_running_loop()
        start = loop.time()
        result = await ToolRetryMiddleware(
            initial_delay=0, hedge_after=0.05
        ).awrap_tool_call(request, handler)
        await asyncio.sleep(0)
        return result, loop.time() - start

    result, elapsed = asyncio.run(run())
    assert result is answer
    assert len(calls) == 2
    assert cancelled == [True]
    assert elapsed < 1


def test_fast_docs_call_is_not_duplicated():
    request = request_for(SEARCH, {"query": "a"})
    answer = ToolMessage(content="docs", tool_call_id="docs-call")
    calls = []

    async def handler(request):
        calls.append(request)
        return answer

    result = asyncio.run(
        ToolRetryMiddleware(initial_delay=0, hedge_after=0.05).awrap_tool_call(
            request, handler
        )
    )
    assert result is answer
    assert len(calls) == 1


def test_other_tools_are_not_hedged():
    request = request_for("check_links", {"urls": []})
    answer = ToolMessage(content="ok", tool_call_id="docs-call")
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.sleep(0.1)
        return answer

    result = asyncio.run(
        ToolRetryMiddleware(initial_delay=0, hedge_after=0.01).awrap_tool_call(
            request, handler
        )
    )
    assert result is answer
    assert len(calls) == 1


def test_hedged_call_waits_for_survivor_when_one_fails():
    request = request_for(SEARCH, {"query": "a"})
    answer = ToolMessage(content="docs", tool_call_id="docs-call")
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            await asyncio.sleep(0.1)
            return answer
        raise RuntimeError("HTTP 500")

    result = asyncio.run(
        ToolRetryMiddleware(initial_delay=0, hedge_after=0.02).awrap_tool_call(
            request, handler
        )
    )
    assert result is answer
