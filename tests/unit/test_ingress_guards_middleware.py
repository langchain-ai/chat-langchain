"""Tests for ingress guards input caps and root-trace metadata helpers."""

from __future__ import annotations

import base64
import io
import os
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from PIL import Image
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware.ingress_guards_middleware import (
    MAX_MESSAGE_CHARS,
    IngressGuardsMiddleware,
    IngressInputMiddleware,
)
from src.utils.trace_root_metadata import build_docs_agent_trace_metadata


def test_before_agent_truncates_oversized_human_message():
    middleware = IngressGuardsMiddleware()
    long_text = "x" * (MAX_MESSAGE_CHARS + 50)
    human = HumanMessage(content=long_text, id="h1")
    state = {"messages": [AIMessage(content="hi"), human]}

    update = middleware.before_agent(state, runtime=SimpleNamespace())

    assert update is not None
    assert update["messages"][0].id == "h1"
    assert len(update["messages"][0].content) == MAX_MESSAGE_CHARS


def test_before_agent_noop_when_under_cap():
    middleware = IngressGuardsMiddleware()
    state = {"messages": [HumanMessage(content="Hello", id="h1")]}

    assert middleware.before_agent(state, runtime=SimpleNamespace()) is None


def test_build_docs_agent_trace_metadata_includes_provenance_and_version(monkeypatch):
    monkeypatch.setenv("LANGCHAIN_REVISION_ID", "rev-a")
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "rev-b")
    monkeypatch.setattr(
        "src.utils.prompt_provenance._USE_LOCAL_PROMPTS",
        True,
    )

    metadata = build_docs_agent_trace_metadata()

    assert metadata["source_type"] == "Chat-LangChain"
    assert metadata["prompt_source"] == "local:instructions.md"
    assert (
        metadata["guardrails_prompt_source"]
        == "local:src/prompts/guardrails_prompts.py"
    )
    assert metadata["LANGSMITH_AGENT_VERSION"] == "rev-a"


def test_build_docs_agent_trace_metadata_falls_back_to_host_revision(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_REVISION_ID", raising=False)
    monkeypatch.setenv("LANGSMITH_HOST_REVISION_ID", "host-rev")

    metadata = build_docs_agent_trace_metadata()
    assert metadata["LANGSMITH_AGENT_VERSION"] == "host-rev"


def _image_url(format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", (1, 1)).save(output, format=format)
    mime = "jpeg" if format == "JPEG" else format.lower()
    return f"data:image/{mime};base64,{base64.b64encode(output.getvalue()).decode()}"


@pytest.mark.parametrize(
    "url",
    [
        "https://internal.example/image.png",
        "ftp://internal.example/image.png",
        "data:application/octet-stream;base64,aGVsbG8=",
        "data:image/png;base64,invalid!",
        "data:image/svg+xml;base64,PHN2Zz4=",
        "data:image/png;base64,"
        + base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\0" * 20).decode(),
    ],
)
def test_unsafe_image_is_replaced(url):
    human = HumanMessage(
        content=[{"type": "image_url", "image_url": {"url": url}}], id="h1"
    )
    update = IngressGuardsMiddleware().before_agent({"messages": [human]}, None)
    assert update["messages"][0].content == [
        {"type": "text", "text": "[unsupported image removed]"}
    ]
    assert update["messages"][0].id == "h1"


@pytest.mark.parametrize("value", [None, 12, "https://example.com/a", {}, {"url": 12}])
def test_malformed_image_is_replaced(value):
    human = HumanMessage(content=[{"type": "image_url", "image_url": value}])
    update = IngressGuardsMiddleware().before_agent({"messages": [human]}, None)
    assert update["messages"][0].content[0]["text"] == "[unsupported image removed]"


@pytest.mark.parametrize("format", ["PNG", "JPEG", "GIF", "WEBP"])
def test_valid_inline_images_are_preserved(format):
    human = HumanMessage(
        content=[{"type": "image_url", "image_url": {"url": _image_url(format)}}]
    )
    assert IngressGuardsMiddleware().before_agent({"messages": [human]}, None) is None


@pytest.mark.parametrize("format", ["PNG", "JPEG", "GIF", "WEBP"])
@pytest.mark.parametrize("missing", [1, 2, 4, 8, 16])
def test_truncated_image_and_mime_mismatch_are_removed(format, missing):
    valid = _image_url(format)
    raw = base64.b64decode(valid.split(",")[1])
    for url in [
        valid.replace(
            "image/" + ("jpeg" if format == "JPEG" else format.lower()),
            "image/png" if format != "PNG" else "image/jpeg",
        ),
        valid.split(",")[0] + "," + base64.b64encode(raw[:-missing]).decode(),
    ]:
        human = HumanMessage(content=[{"type": "image_url", "image_url": {"url": url}}])
        update = IngressGuardsMiddleware().before_agent({"messages": [human]}, None)
        assert update["messages"][0].content[0]["text"] == "[unsupported image removed]"


def test_historical_images_are_repaired_in_checkpoint():
    builder = StateGraph(MessagesState)
    builder.add_node(
        "guard", lambda state: IngressGuardsMiddleware().before_agent(state, None) or {}
    )
    builder.add_edge(START, "guard")
    builder.add_edge("guard", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "recovery"}}
    graph.update_state(
        config,
        {
            "messages": [
                HumanMessage(
                    id="old",
                    content=[
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://internal.example/a"},
                        }
                    ],
                )
            ]
        },
    )
    result = graph.invoke(
        {"messages": [HumanMessage(id="new", content="hello")]}, config
    )
    assert len(result["messages"]) == 2
    history = graph.get_state(config).values["messages"]
    assert history[0].id == "old"
    assert history[0].content == [
        {"type": "text", "text": "[unsupported image removed]"}
    ]
    assert history[1].content == "hello"
    assert (
        graph.invoke({"messages": [HumanMessage(content="next")]}, config)["messages"][
            0
        ]
        == history[0]
    )


def _input_client():
    async def echo(request):
        return JSONResponse(await request.json())

    app = Starlette(
        routes=[
            Route(path, echo, methods=["POST"])
            for path in [
                "/runs",
                "/runs/stream",
                "/runs/wait",
                "/runs/batch",
                "/threads/t/runs/stream",
                "/threads/t/state",
            ]
        ]
    )
    app.add_middleware(IngressInputMiddleware)
    return TestClient(app)


@pytest.mark.parametrize(
    "messages",
    [
        12345,
        {},
        "hi",
        None,
        [123],
        [{}],
        [{"role": "invalid", "content": "hi"}],
        [{"role": "user", "content": 1}],
    ],
)
@pytest.mark.parametrize(
    "path", ["/runs", "/runs/stream", "/runs/wait", "/threads/t/runs/stream"]
)
def test_http_boundary_rejects_bad_messages(messages, path):
    response = _input_client().post(path, json={"input": {"messages": messages}})
    assert response.status_code == 422
    assert response.json()["detail"]


def test_http_boundary_sanitizes_before_state_write():
    for path, key in [("/runs", "input"), ("/threads/t/state", "values")]:
        response = _input_client().post(
            path,
            json={
                key: {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": "https://internal.example/a"},
                                }
                            ],
                        }
                    ]
                }
            },
        )
        assert response.status_code == 200
        assert response.json()[key]["messages"][0]["content"] == [
            {"type": "text", "text": "[unsupported image removed]"}
        ]


def test_http_boundary_validates_batch_and_recursion_limit():
    assert (
        _input_client()
        .post("/runs/batch", json=[{"input": {"messages": 123}}])
        .status_code
        == 422
    )
    assert (
        _input_client()
        .post(
            "/runs", json={"input": {"messages": []}, "config": {"recursion_limit": 0}}
        )
        .status_code
        == 422
    )
    assert _input_client().post("/runs", json={"input": None}).status_code == 200
