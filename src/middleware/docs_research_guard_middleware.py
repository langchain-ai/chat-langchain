"""Ensure substantive technical answers use fresh documentation research."""

from __future__ import annotations

import os
import re
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

SEARCH_TOOLS = frozenset(
    {
        "search_docs_by_lang_chain",
        "search_support_articles",
    }
)
READ_TOOLS = frozenset(
    {
        "query_docs_filesystem_docs_by_lang_chain",
        "get_support_article_content",
        "read_file",
        "fetch_langchain_pricing",
    }
)
RESEARCH_TOOLS = SEARCH_TOOLS | READ_TOOLS
RESEARCH_GUARD_DISABLED_ENV = "DOCS_RESEARCH_GUARD_DISABLED"
_RETRY_INSTRUCTIONS = (
    "Before answering, research this question on this turn. Call "
    "search_docs_by_lang_chain and query_docs_filesystem_docs_by_lang_chain, "
    "then use the retrieved documentation to answer. Do not answer from memory."
)
_DISCLOSURE = (
    "Documentation could not be consulted on this turn, so the following answer "
    "may contain unverified information."
)
_MAX_FORCED_ATTEMPTS = 2
_DOCS_URL_PATTERN = re.compile(r"https://docs\.langchain\.com/[^\s<>\]\)\"']+")
_CODE_BLOCK_PATTERN = re.compile(r"```.*?(?:```|$)", re.DOTALL)
_LARGE_RESULT_POINTER_PATTERN = re.compile(r"^/large_tool_results/[^\s]+$")
_NONTECHNICAL_USER_TURN_PATTERN = re.compile(
    r"(?:hi|hello|hey|good\s+(?:morning|afternoon|evening)|hola|bonjour|salut|"
    r"你好|您好|こんにちは|こんばんは|привет|здравствуйте|what\s+can\s+you\s+do|"
    r"who\s+are\s+you|(?:can\s+you\s+)?help(?:\s+me)?)[!.?,\s]*",
    re.IGNORECASE,
)
_TECHNICAL_USER_SIGNAL_PATTERN = re.compile(
    r"```|`[^`]+`|https?://|\b(?:error|exception|traceback|stack\s+trace)\b|"
    r"\b(?:langchain|langgraph|langsmith|fleet|deepagents)\b",
    re.IGNORECASE,
)
_TECHNICAL_IDENTIFIER_PATTERN = re.compile(
    r"\b(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*\b|"
    r"\b[A-Za-z]+_[A-Za-z0-9_]+\b|"
    r"\b(?!(?:LangChain|LangGraph|LangSmith|DeepAgents)\b)"
    r"[A-Z][a-z]+[A-Z][A-Za-z0-9]*\b|"
    r"\b(?:from\s+[\w.]+\s+import|import\s+\w+)\b|"
    r"(?:^|\s)(?:\$\s*)?(?:python(?:3)?|pip|uv|npm|pnpm|poetry|git|curl)\s+\S+",
    re.MULTILINE,
)
_FORCED_TURN_KEY = "_docs_research_guard_forced_turn"
_FORCED_ATTEMPTS_KEY = "_docs_research_guard_forced_attempts"


class DocsResearchGuardState(AgentState, total=False):
    """State persisted by the documentation research guard."""

    _docs_research_guard_forced_turn: str
    _docs_research_guard_forced_attempts: int


class DocsResearchGuardMiddleware(AgentMiddleware):
    """Force fresh documentation research before terminal technical answers."""

    state_schema = DocsResearchGuardState

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Require fresh research before returning a technical answer."""
        response = await handler(request)
        state = self._state_for_turn(request)
        if not self._should_retry(request, response):
            self._clear_attempts(state)
            return response

        turn_key = self._turn_key(request.messages)
        while self._attempt_count(state, turn_key) < _MAX_FORCED_ATTEMPTS:
            self._record_attempt(state, turn_key)
            retry_request = request.override(
                messages=[
                    *request.messages,
                    HumanMessage(content=_RETRY_INSTRUCTIONS),
                ],
                system_message=self._retry_system_message(request),
                tool_choice={
                    "type": "function",
                    "function": {"name": "search_docs_by_lang_chain"},
                },
            )
            response = await handler(retry_request)
            if self._has_pending_tool_calls(self._response_messages(response)):
                return response
            if self._has_research_tool(
                self._turn_messages(request.messages, self._response_messages(response))
            ):
                self._clear_attempts(state)
                return response
            if not self._is_substantive_technical_answer(
                self._response_messages(response)
            ):
                self._clear_attempts(state)
                return response

        self._clear_attempts(state)
        return self._sanitize_response(request, response)

    def _should_retry(self, request: ModelRequest, response: ModelResponse) -> bool:
        if os.getenv(RESEARCH_GUARD_DISABLED_ENV, "").lower() in {"1", "true", "yes"}:
            return False
        messages = request.messages
        latest_human_index = self._latest_human_index(messages)
        if latest_human_index < 0:
            return False
        if not self._user_turn_has_technical_signal(messages[latest_human_index]):
            return False
        response_messages = self._response_messages(response)
        if self._has_pending_tool_calls(response_messages):
            return False
        current_turn = self._turn_messages(messages, response_messages)
        if self._has_research_tool(current_turn):
            return False
        return self._is_substantive_technical_answer(response_messages)

    def _latest_human_index(self, messages: list[BaseMessage]) -> int:
        for index in range(len(messages) - 1, -1, -1):
            if getattr(messages[index], "type", None) == "human":
                return index
        return -1

    def _turn_key(self, messages: list[BaseMessage]) -> str:
        index = self._latest_human_index(messages)
        human = messages[index]
        return str(getattr(human, "id", None) or f"{index}:{human.content!r}")

    def _turn_messages(
        self, request_messages: list[BaseMessage], response_messages: list[BaseMessage]
    ) -> list[BaseMessage]:
        latest_human_index = self._latest_human_index(request_messages)
        return [
            *request_messages[latest_human_index + 1 :],
            *response_messages,
        ]

    def _response_messages(self, response: ModelResponse) -> list[BaseMessage]:
        result = getattr(response, "result", None)
        if result is not None:
            return list(result)
        return [response]

    def _has_pending_tool_calls(self, messages: list[BaseMessage]) -> bool:
        return any(
            isinstance(message, AIMessage) and bool(message.tool_calls)
            for message in messages
        )

    def _has_research_tool(self, messages: list[BaseMessage]) -> bool:
        tool_messages = [
            message
            for message in messages
            if isinstance(message, ToolMessage)
            and message.name in RESEARCH_TOOLS
            and self._has_usable_content(message)
        ]
        for index, message in enumerate(tool_messages):
            if not self._is_large_result_pointer(message):
                return True
            if any(
                later.name == "read_file" and not self._is_large_result_pointer(later)
                for later in tool_messages[index + 1 :]
            ):
                return True
        return False

    def _has_usable_content(self, message: ToolMessage) -> bool:
        return message.status not in {"error", "failure", "failed"} and bool(
            self._message_text(message).strip()
        )

    def _is_large_result_pointer(self, message: ToolMessage) -> bool:
        return bool(
            _LARGE_RESULT_POINTER_PATTERN.fullmatch(self._message_text(message).strip())
        )

    def _is_substantive_technical_answer(self, messages: list[BaseMessage]) -> bool:
        text = "\n".join(self._message_text(message) for message in messages)
        if len(text.strip()) < 40:
            return False
        return bool(
            "```" in text
            or re.search(r"`[^`]+`", text)
            or _TECHNICAL_IDENTIFIER_PATTERN.search(text)
            or re.search(
                r"\b(?:api|class|function|method|constructor|parameter|argument|"
                r"config(?:uration)?|option|property|field|tool call|invoke|returns?)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _user_turn_has_technical_signal(self, message: BaseMessage) -> bool:
        text = self._message_text(message).strip()
        if _TECHNICAL_USER_SIGNAL_PATTERN.search(text):
            return True
        return len(text) > 120 or not _NONTECHNICAL_USER_TURN_PATTERN.fullmatch(text)

    def _message_text(self, message: BaseMessage) -> str:
        content: Any = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(content)

    def _retry_system_message(self, request: ModelRequest) -> SystemMessage:
        existing = request.system_message.text if request.system_message else ""
        content = f"{existing}\n\n{_RETRY_INSTRUCTIONS}".strip()
        return SystemMessage(content=content)

    def _state_for_turn(self, request: ModelRequest) -> dict[str, Any]:
        state = request.state if isinstance(request.state, dict) else {}
        turn_key = self._turn_key(request.messages)
        if state.get(_FORCED_TURN_KEY) != turn_key:
            state[_FORCED_TURN_KEY] = turn_key
            state[_FORCED_ATTEMPTS_KEY] = 0
        return state

    def _attempt_count(self, state: dict[str, Any], turn_key: str) -> int:
        if state.get(_FORCED_TURN_KEY) != turn_key:
            return 0
        return int(state.get(_FORCED_ATTEMPTS_KEY, 0))

    def _record_attempt(self, state: dict[str, Any], turn_key: str) -> None:
        state[_FORCED_TURN_KEY] = turn_key
        state[_FORCED_ATTEMPTS_KEY] = self._attempt_count(state, turn_key) + 1

    def _clear_attempts(self, state: dict[str, Any]) -> None:
        state[_FORCED_ATTEMPTS_KEY] = 0

    def _sanitize_response(
        self, request: ModelRequest, response: ModelResponse
    ) -> ModelResponse:
        current_turn = self._turn_messages(request.messages, [])
        grounded_urls = {
            url.rstrip(".,;:")
            for message in current_turn
            if isinstance(message, ToolMessage) and message.name in RESEARCH_TOOLS
            for url in _DOCS_URL_PATTERN.findall(self._message_text(message))
        }
        messages = list(response.result)
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if not isinstance(message, AIMessage):
                continue
            text = _CODE_BLOCK_PATTERN.sub("", self._message_text(message))
            text = _DOCS_URL_PATTERN.sub(
                lambda match: (
                    match.group(0)
                    if match.group(0).rstrip(".,;:") in grounded_urls
                    else ""
                ),
                text,
            )
            messages[index] = message.model_copy(
                update={"content": f"{_DISCLOSURE}\n\n{text.strip()}"}
            )
            break
        return ModelResponse(
            result=messages,
            structured_response=response.structured_response,
        )


__all__ = [
    "DocsResearchGuardMiddleware",
    "READ_TOOLS",
    "RESEARCH_TOOLS",
    "SEARCH_TOOLS",
]
