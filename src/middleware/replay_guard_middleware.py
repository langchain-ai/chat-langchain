"""Middleware that regenerates near-duplicate answers on changed follow-up turns."""

import difflib
import logging
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class ReplayGuardMiddleware(AgentMiddleware):
    """Regenerate a replayed answer when the latest question changed."""

    def _message_text(self, message: object) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content
        return str(content)

    def _response_text(self, response: ModelCallResult) -> str:
        if isinstance(response, AIMessage):
            return self._message_text(response)
        result = getattr(response, "result", [])
        if not result:
            return ""
        return self._message_text(result[0])

    def _retry_request(self, request: ModelRequest) -> ModelRequest | None:
        human_messages = [
            message for message in request.messages if isinstance(message, HumanMessage)
        ]
        ai_messages = [
            message for message in request.messages if isinstance(message, AIMessage)
        ]
        if len(human_messages) < 2 or not ai_messages:
            return None

        latest_human = self._message_text(human_messages[-1])
        previous_human = self._message_text(human_messages[-2])
        if difflib.SequenceMatcher(None, latest_human, previous_human).ratio() >= 0.9:
            return None

        instruction = (
            "Answer the latest user question below. Do not reuse or reproduce the "
            "previous assistant answer. Latest user question:\n"
            f"{latest_human}"
        )
        system_text = request.system_message.content if request.system_message else ""
        system_message = SystemMessage(
            content=f"{system_text}\n\n{instruction}" if system_text else instruction
        )
        return request.override(system_message=system_message)

    def _needs_retry(self, request: ModelRequest, response: ModelCallResult) -> bool:
        retry_request = self._retry_request(request)
        if retry_request is None:
            return False
        ai_messages = [
            message for message in request.messages if isinstance(message, AIMessage)
        ]
        previous_answer = self._message_text(ai_messages[-1])
        response_text = self._response_text(response)
        return (
            bool(response_text)
            and difflib.SequenceMatcher(None, response_text, previous_answer).ratio()
            > 0.95
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Regenerate a near-duplicate answer for a changed question."""
        response = handler(request)
        retry_request = self._retry_request(request)
        if retry_request is not None and self._needs_retry(request, response):
            logger.warning("Replay guard regenerated a near-duplicate answer")
            return handler(retry_request)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Regenerate a near-duplicate answer asynchronously."""
        response = await handler(request)
        retry_request = self._retry_request(request)
        if retry_request is not None and self._needs_retry(request, response):
            logger.warning("Replay guard regenerated a near-duplicate answer")
            return await handler(retry_request)
        return response
