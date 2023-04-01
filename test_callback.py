from unittest.mock import AsyncMock

import pytest

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler


@pytest.fixture
def websocket():
    return AsyncMock()


@pytest.mark.asyncio
async def test_llm_new_token(websocket):
    handler = StreamingLLMCallbackHandler(websocket)
    token = "test_token"
    expected_response = {
        "sender": "bot",
        "message": token,
        "type": "stream",
        "sources": None,
    }
    await handler.on_llm_new_token(token)
    websocket.send_json.assert_called_once_with(expected_response)


@pytest.mark.asyncio
async def test_llm_start(websocket):
    handler = QuestionGenCallbackHandler(websocket)
    serialized = {"key1": "val1"}
    prompts = ["prompts1", "prompts2"]
    expected_response = {
        "sender": "bot",
        "message": "Synthesizing question...",
        "type": "info",
        "sources": None,
    }
    await handler.on_llm_start(serialized, prompts)
    websocket.send_json.assert_called_once_with(expected_response)
    websocket.send_json.assert_called_once_with(expected_response)
