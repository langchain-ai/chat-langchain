# Footer enforcement middleware for the docs agent.
import logging
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

logger = logging.getLogger(__name__)

# Localized variants of the "Relevant docs:" footer label. Keep in sync with
# the variant set used by the localized-label footer middleware.
FOOTER_LABELS = (
    "**relevant docs:**",
    "相关文档",
    "相關文檔",
    "関連ドキュメント",
    "관련 문서",
    "documentos relevantes",
    "documentación relevante",
    "documentation pertinente",
    "relevante dokumente",
    "tài liệu liên quan",
    "соответствующая документация",
)

# Minimum response length (in characters) before we enforce the footer. Short
# replies (greetings, clarifications, scope refusals) are exempt.
MIN_LENGTH_FOR_FOOTER = 600

# Window at the end of the message we scan for any footer label.
FOOTER_SCAN_WINDOW = 800

CORRECTION_INSTRUCTION = (
    "Your previous answer was missing the mandatory `**Relevant docs:**` footer. "
    "Re-emit the SAME answer in full, appending a `**Relevant docs:**` block "
    "(or the language-appropriate localized label) at the very end with at "
    "least one `[text](url)` entry pointing to the URLs visible in this turn's "
    "tool results. Do not add anything after the footer."
)


def _extract_text(message: AIMessage) -> str:
    """Flatten an AIMessage's content into plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _turn_called_tools(request: ModelRequest) -> bool:
    """Return True if a ToolMessage appears since the last HumanMessage in this turn."""
    for message in reversed(request.messages):
        if isinstance(message, HumanMessage):
            return False
        if isinstance(message, ToolMessage):
            return True
    return False


def _missing_footer(text: str) -> bool:
    tail = text[-FOOTER_SCAN_WINDOW:].lower()
    return not any(label in tail for label in FOOTER_LABELS)


class FooterEnforcementMiddleware(AgentMiddleware):
    """Retries a tool-backed response when the mandatory footer is missing."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        response = await handler(request)

        if not _turn_called_tools(request):
            return response

        ai_message = next(
            (m for m in reversed(response.result) if isinstance(m, AIMessage)),
            None,
        )
        if ai_message is None or ai_message.tool_calls:
            return response

        text = _extract_text(ai_message)
        if len(text) <= MIN_LENGTH_FOR_FOOTER or not _missing_footer(text):
            return response

        logger.warning(
            "Docs agent response missing Relevant docs footer; issuing corrective retry"
        )

        correction = HumanMessage(content=CORRECTION_INSTRUCTION)
        retry_request = request.override(
            messages=[*request.messages, ai_message, correction]
        )
        return await handler(retry_request)


__all__ = ["FooterEnforcementMiddleware"]
