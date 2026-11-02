"""Evaluation deployments can disable writes without disabling guardrails."""

import asyncio
import os
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import guardrails_middleware as guardrails_module


@pytest.mark.parametrize("decision", ["ALLOWED", "BLOCKED"])
@pytest.mark.parametrize("collect_samples", [True, False])
def test_sampling_switch_preserves_classification(
    monkeypatch, decision, collect_samples
):
    monkeypatch.setattr(guardrails_module, "init_chat_model", Mock())
    middleware = guardrails_module.GuardrailsMiddleware(
        model="test", fallback_model="test", collect_samples=collect_samples
    )
    classify = AsyncMock(return_value={"decision": decision, "explanation": "test"})
    monkeypatch.setattr(middleware, "_classify_query", classify)
    monkeypatch.setattr(
        middleware,
        "_generate_rejection_message",
        AsyncMock(return_value=AIMessage(content="Blocked")),
    )
    write = AsyncMock()
    monkeypatch.setattr(middleware, "_add_to_dataset", write)
    monkeypatch.setattr(guardrails_module.random, "random", lambda: 0)

    async def invoke():
        result = await middleware.abefore_agent(
            {"messages": [HumanMessage(content="test query")]}, Runtime(context=None)
        )
        await asyncio.sleep(0)  # Let any scheduled sample write run.
        return result

    result = asyncio.run(invoke())
    classify.assert_awaited_once()
    assert write.await_count == int(collect_samples)
    if decision == "BLOCKED":
        assert result["off_topic_query"] is True
        assert result["jump_to"] == "end"
    else:
        assert result is None


def test_disabled_sample_writer_never_opens_client(monkeypatch):
    monkeypatch.setattr(guardrails_module, "init_chat_model", Mock())
    middleware = guardrails_module.GuardrailsMiddleware(
        model="test", fallback_model="test", collect_samples=False
    )
    client = Mock(side_effect=AssertionError("Dataset client must not be created"))
    monkeypatch.setattr(guardrails_module.ls, "AsyncClient", client)
    asyncio.run(middleware._add_to_dataset("query", "BLOCKED", "reason", "preview"))
    client.assert_not_called()


def test_sample_collection_enabled_by_default(monkeypatch):
    monkeypatch.setattr(guardrails_module, "init_chat_model", Mock())
    middleware = guardrails_module.GuardrailsMiddleware(
        model="test", fallback_model="test"
    )
    assert middleware.collect_samples is True


@pytest.mark.parametrize(
    "value,excluded", [(None, []), ("false", []), ("true", ["submit_feedback"])]
)
def test_feedback_declaration(monkeypatch, value, excluded):
    if value is None:
        monkeypatch.delenv("CLC_DISABLE_SUBMIT_FEEDBACK", raising=False)
    else:
        monkeypatch.setenv("CLC_DISABLE_SUBMIT_FEEDBACK", value)
    # Capture the authored declaration without requiring the managed runtime.
    managed = types.ModuleType("managed_deepagents")
    managed.define_mcp = lambda **config: config
    monkeypatch.setitem(sys.modules, "managed_deepagents", managed)
    module = runpy.run_path(str(Path(__file__).parents[2] / "tools" / "mcp.py"))
    server = module["mcp"]["servers"]["langchainDocs"]
    assert server["url"] == "https://docs.langchain.com/mcp"
    assert server["exclude_tools"] == excluded
