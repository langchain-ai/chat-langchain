"""Build model retry requests with user-final message turns."""

from __future__ import annotations

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


def build_retry_request(
    request: ModelRequest,
    response: ModelResponse,
    instructions: str,
) -> ModelRequest:
    """Build a retry request that ends with a human instruction."""
    prior_draft = "\n\n".join(
        _message_text(message) for message in _response_messages(response)
    )
    retry_content = f"{instructions}\n\nPrior draft (quoted):\n---\n{prior_draft}\n---"
    existing = request.system_message.text if request.system_message else ""
    system_message = SystemMessage(content=f"{existing}\n\n{instructions}".strip())
    return request.override(
        messages=[*request.messages, HumanMessage(content=retry_content)],
        system_message=system_message,
    )


def _response_messages(response: ModelResponse) -> list[BaseMessage]:
    result = getattr(response, "result", None)
    return list(result) if result is not None else [response]


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)
