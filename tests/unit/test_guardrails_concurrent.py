"""Guardrails classifier runs alongside the first model call."""

import asyncio
import os
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import langsmith as ls
import pytest
from langchain.agents.middleware import ModelResponse
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.runtime import Runtime

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.guardrails_middleware import (  # noqa: E402
    GuardrailsClassificationError,
    GuardrailsMiddleware,
)


def _middleware(decide) -> GuardrailsMiddleware:
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.classifier_llms = []
    middleware.block_off_topic = True
    middleware.concurrent = True
    middleware._pending = {}
    middleware._decide = decide
    return middleware


def _request(messages):
    return SimpleNamespace(messages=messages)


async def _run_turn(middleware, model_delay=0.2, verdict_delay=0.2):
    messages = [HumanMessage(content="How do agents work?", id="h1")]
    calls = {"model_started": 0, "model_finished": 0, "tags": None}

    async def handler(request):  # noqa: ARG001
        calls["model_started"] += 1
        calls["tags"] = (var_child_runnable_config.get() or {}).get("tags")
        await asyncio.sleep(model_delay)
        calls["model_finished"] += 1
        return ModelResponse(result=[AIMessage(content="model answer")])

    before = await middleware.abefore_agent(
        {"messages": messages}, Runtime(context=None)
    )
    assert before is None
    start = asyncio.get_running_loop().time()
    result = await middleware.awrap_model_call(_request(messages), handler)
    elapsed = asyncio.get_running_loop().time() - start
    return result, calls, elapsed


def test_allowed_runs_classifier_and_model_concurrently():
    async def decide(state):  # noqa: ARG001
        await asyncio.sleep(0.2)
        return None

    middleware = _middleware(decide)
    result, calls, elapsed = asyncio.run(_run_turn(middleware))

    assert result.result[0].content == "model answer"
    assert calls["model_finished"] == 1
    assert "nostream" in calls["tags"]
    assert elapsed < 0.35  # concurrent, not 0.4s serial
    assert middleware._pending == {}


def test_blocked_returns_rejection_and_discards_model_output():
    rejection = AIMessage(content="Please ask about LangChain.")

    async def decide(state):  # noqa: ARG001
        await asyncio.sleep(0.05)
        return {"messages": [rejection], "off_topic_query": True, "jump_to": "end"}

    middleware = _middleware(decide)
    result, calls, _ = asyncio.run(_run_turn(middleware, model_delay=0.5))

    assert result.result == [rejection]
    assert calls["model_finished"] == 0  # cancelled before it finished
    assert not result.result[0].tool_calls


def test_classifier_error_allows():
    async def decide(state):  # noqa: ARG001
        raise GuardrailsClassificationError("down")

    middleware = _middleware(decide)
    result, calls, _ = asyncio.run(_run_turn(middleware))

    assert result.result[0].content == "model answer"
    assert calls["model_finished"] == 1


def test_later_model_calls_pass_straight_through_and_stream():
    async def decide(state):  # noqa: ARG001
        return None

    middleware = _middleware(decide)

    async def go():
        messages = [HumanMessage(content="q", id="h1")]
        await middleware.abefore_agent({"messages": messages}, Runtime(context=None))
        tags_seen = []

        async def handler(request):  # noqa: ARG001
            tags_seen.append((var_child_runnable_config.get() or {}).get("tags") or [])
            return ModelResponse(result=[AIMessage(content="x")])

        await middleware.awrap_model_call(_request(messages), handler)
        await middleware.awrap_model_call(_request(messages), handler)
        return tags_seen

    tags_seen = asyncio.run(go())
    assert "nostream" in tags_seen[0]
    assert "nostream" not in tags_seen[1]


def test_non_concurrent_mode_still_blocks_in_before_agent():
    rejection = AIMessage(content="no")

    async def decide(state):  # noqa: ARG001
        return {"messages": [rejection], "off_topic_query": True, "jump_to": "end"}

    middleware = _middleware(decide)
    middleware.concurrent = False
    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="Write a novel.")]},
            Runtime(context=None),
        )
    )
    assert result["jump_to"] == "end"


@pytest.mark.parametrize("decision", ["ALLOWED", "BLOCKED"])
@pytest.mark.parametrize("concurrent", [True, False])
def test_decision_metadata_persisted_on_open_spans(monkeypatch, decision, concurrent):
    from src.middleware import guardrails_middleware as guardrails_module

    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.block_off_topic = True
    middleware.concurrent = concurrent
    middleware._pending = {}
    monkeypatch.setattr(guardrails_module, "ALLOWED_SAMPLE_RATE", 0)
    monkeypatch.setattr(middleware, "_add_to_dataset", AsyncMock())
    explanation = "Classification explanation."
    rejection = AIMessage(content="Please ask about LangChain.")
    started = asyncio.Event()
    release = asyncio.Event()
    children = []
    parents = []

    async def classify(messages):
        parents.append(ls.get_current_run_tree())
        with ls.trace("classifier-model", run_type="llm") as run:
            children.append(run)
            started.set()
            if concurrent:
                await release.wait()
            return {"decision": decision, "explanation": explanation}

    async def reject(content):
        with ls.trace("rejection-model", run_type="llm") as run:
            children.append(run)
            return rejection

    monkeypatch.setattr(middleware, "_classify_query", classify)
    monkeypatch.setattr(middleware, "_generate_rejection_message", reject)
    persisted = {}
    client = Mock(spec=ls.Client)

    def save(run_id, **kwargs):
        persisted[run_id] = deepcopy(kwargs)

    client.update_run.side_effect = save

    async def go():
        messages = [HumanMessage(content="query", id="metadata-turn")]
        with ls.tracing_context(enabled=True, client=client):
            with ls.trace("before-agent") as before_run:
                before = await middleware.abefore_agent(
                    {"messages": messages}, Runtime(context=None)
                )
                if concurrent:
                    await started.wait()
            with ls.trace("model-call") as model_run:
                release.set()

                async def handler(request):
                    return ModelResponse(result=[AIMessage(content="model answer")])

                result = await middleware.awrap_model_call(_request(messages), handler)
        return before, result, before_run, model_run

    before, result, before_run, model_run = asyncio.run(go())
    if concurrent:
        classifier_run = parents[0]
        assert classifier_run.name == "GuardrailsClassifier"
        for child in children:
            assert child.parent_run_id == classifier_run.id
            assert classifier_run.start_time <= child.start_time
            assert child.end_time <= classifier_run.end_time
        assert children[0].end_time > before_run.end_time
        decision_runs = [classifier_run, model_run]
        assert result.result == (
            [rejection]
            if decision == "BLOCKED"
            else [AIMessage(content="model answer")]
        )
    else:
        decision_runs = [before_run]
        assert before == (
            {"messages": [rejection], "off_topic_query": True, "jump_to": "end"}
            if decision == "BLOCKED"
            else None
        )
    for run in decision_runs:
        metadata = persisted[run.id]["extra"]["metadata"]
        assert metadata["guardrails_result"] == decision
        assert metadata["guardrails_explanation"] == explanation
