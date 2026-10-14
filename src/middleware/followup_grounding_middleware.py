"""Require content reads before answering questions naming ungrounded APIs."""

import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import replace

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

CONTENT_READ_TOOLS = {
    "query_docs_filesystem_docs_by_lang_chain",
    "get_support_article_content",
}
RETRY_MARKER = "followup_grounding_retry"
UNVERIFIED_RESPONSE = (
    "I couldn't verify this API in the available documentation or support content, "
    "so I can't give a grounded technical answer."
)
_API_TOKEN = re.compile(r"\b[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)*\b")
_CAMEL_CASE = re.compile(r"(?:[A-Z][a-z0-9]+|[A-Z]+(?=[A-Z][a-z])){2,}[A-Z]*\Z")
_SNAKE_CASE = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+\Z")
_READ_FAILURE = re.compile(
    r"^(?:error\b|unexpected error\b|no results? found\b|no content available\b)"
    r"|(?:no such file or directory|command not found)"
    r"|^Article ID .+ not found in knowledge base\.",
    re.IGNORECASE,
)


def extract_api_identifiers(text: str) -> set[str]:
    """Extract CamelCase, snake_case, and dotted API identifiers."""
    return {
        token
        for token in _API_TOKEN.findall(text)
        if "." in token or _CAMEL_CASE.fullmatch(token) or _SNAKE_CASE.fullmatch(token)
    }


def successful_content_read(message: object) -> bool:
    """Recognize nonempty content reads, excluding tool and source errors."""
    if not isinstance(message, ToolMessage):
        return False
    if message.name not in CONTENT_READ_TOOLS or message.status == "error":
        return False
    text = message.text.strip()
    if not text or _READ_FAILURE.search(text):
        return False
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, (dict, list)):
        if not payload:
            return False
        if isinstance(payload, dict) and (
            payload.get("error")
            or payload.get("isError")
            or ("content" in payload and not payload["content"])
        ):
            return False
    if message.name == "get_support_article_content" and "Content:" in text:
        content = text.split("Content:", 1)[1].strip()
        return bool(content) and not bool(_READ_FAILURE.search(content))
    return True


class FollowupGroundingMiddleware(AgentMiddleware):
    """Withhold unsupported API answers and persist one retry in turn messages."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Retry an ungrounded final answer once, then fail closed."""
        response = await handler(request)
        if not response.result or not isinstance(response.result[-1], AIMessage):
            return response
        if any(
            isinstance(message, AIMessage) and message.tool_calls
            for message in response.result
        ):
            return response

        messages = request.state["messages"]
        human_index = next(
            (
                index
                for index in range(len(messages) - 1, -1, -1)
                if isinstance(messages[index], HumanMessage)
            ),
            None,
        )
        if human_index is None:
            return response
        identifiers = extract_api_identifiers(messages[human_index].text)
        prior_identifiers = set().union(
            *(
                extract_api_identifiers(message.text)
                for message in messages[:human_index]
                if successful_content_read(message)
            )
        )
        new_identifiers = identifiers - prior_identifiers
        turn_messages = messages[human_index + 1 :]
        if not new_identifiers or any(
            successful_content_read(message) for message in turn_messages
        ):
            return response

        retry_used = any(
            isinstance(message, AIMessage)
            and message.additional_kwargs.get(RETRY_MARKER)
            for message in turn_messages
        )
        if not retry_used:
            instruction = SystemMessage(
                content=(
                    "Before answering this turn, search with search_docs_by_lang_chain "
                    "for these unverified APIs: "
                    + ", ".join(sorted(new_identifiers))
                    + ". Then read relevant pages with "
                    "query_docs_filesystem_docs_by_lang_chain. Search results and "
                    "check_links are not content reads. If relevant content cannot be "
                    "read, acknowledge that you cannot verify the answer."
                )
            )
            response = await handler(
                request.override(messages=[*request.messages, instruction])
            )
            if any(
                isinstance(message, AIMessage) and message.tool_calls
                for message in response.result
            ):
                return replace(
                    response,
                    result=[
                        message.model_copy(
                            update={
                                "additional_kwargs": {
                                    **message.additional_kwargs,
                                    RETRY_MARKER: True,
                                }
                            }
                        )
                        if isinstance(message, AIMessage)
                        else message
                        for message in response.result
                    ],
                )

        return ModelResponse(
            result=[
                AIMessage(
                    content=UNVERIFIED_RESPONSE,
                    additional_kwargs={RETRY_MARKER: True},
                )
            ]
        )
