"""Tests for follow-up answer replay protection."""

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from src.middleware.replay_guard_middleware import ReplayGuardMiddleware


def _request(latest_question: str) -> ModelRequest:
    return ModelRequest(
        model=object(),
        messages=[
            HumanMessage(content="How do I configure tracing?"),
            AIMessage(
                content="Tracing is configured with the tracing environment variables."
            ),
            HumanMessage(content=latest_question),
        ],
    )


def test_replay_guard_regenerates_changed_follow_up_once():
    middleware = ReplayGuardMiddleware()
    calls = []

    def handler(request):
        calls.append(request)
        answer = (
            "Tracing is configured with the tracing environment variables."
            if len(calls) == 1
            else "Deploy with the agent deployment workflow."
        )
        return ModelResponse(result=[AIMessage(content=answer)])

    response = middleware.wrap_model_call(
        _request("How do I deploy an agent?"), handler
    )

    assert len(calls) == 2
    assert "How do I deploy an agent?" in calls[1].system_message.content
    assert response.result[0].content == "Deploy with the agent deployment workflow."


def test_replay_guard_allows_repeated_question():
    middleware = ReplayGuardMiddleware()
    calls = []

    def handler(request):
        calls.append(request)
        return ModelResponse(
            result=[
                AIMessage(
                    content="Tracing is configured with the tracing environment variables."
                )
            ]
        )

    middleware.wrap_model_call(_request("How do I configure tracing?"), handler)

    assert len(calls) == 1
