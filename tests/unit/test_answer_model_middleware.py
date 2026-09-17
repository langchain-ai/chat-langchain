import asyncio
import os
from unittest.mock import MagicMock

os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("ANTHROPIC_API_KEY", "test")
os.environ.setdefault("USE_LOCAL_PROMPTS", "1")

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import config


def _fake_model(model_name: str) -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.model = model_name
    return model


def test_fallback_records_only_successful_model(monkeypatch):
    primary = _fake_model("gemini-3.5-flash-lite")
    fallback = _fake_model("claude-haiku-4-5-20251001")
    middleware = config.AnswerModelFallbackMiddleware(fallback)
    recorded: list[dict[str, str]] = []
    monkeypatch.setattr(
        config, "set_root_metadata", lambda **values: recorded.append(values)
    )
    calls = 0

    async def handler(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("primary unavailable")
        return ModelResponse(result=[AIMessage(content="fallback answer")])

    result = asyncio.run(
        middleware.awrap_model_call(
            ModelRequest(model=primary, messages=[HumanMessage(content="Hi")]),
            handler,
        )
    )

    assert result.result[0].content == "fallback answer"
    assert recorded == [{"answer_model": "claude-haiku-4.5"}]
