"""Force a doc-search tool call when the user's latest message has API-shaped signals."""
import logging
import re
from typing import Awaitable, Callable

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage

logger = logging.getLogger(__name__)

DOC_SEARCH_TOOLS = (
    "search_docs_by_lang_chain",
    "query_docs_filesystem_docs_by_lang_chain",
)
FORCED_TOOL_NAME = "search_docs_by_lang_chain"

# Treat these phrasings as direct asks for an API surface fact.
_API_QUESTION_PATTERNS = (
    re.compile(r"\bvalid (values?|options?|modes?)\s+(for|of)\b", re.IGNORECASE),
    re.compile(r"\bdefaults?\s+(value|for|of)\b", re.IGNORECASE),
    re.compile(r"\b(allowed|accepted|supported)\s+(values?|options?|modes?)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(does|is)\s+the\s+\w+\s+(parameter|argument|option|field)\b", re.IGNORECASE),
)

# camelCase / PascalCase identifiers that look like framework API names
# (>=2 letters, an internal lowercase->uppercase boundary, no spaces).
_CAMEL_CASE_RE = re.compile(r"\b[a-z]+[A-Z][A-Za-z0-9]+\b")
_PASCAL_CASE_RE = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]+){1,}\b")
_CODE_FENCE_RE = re.compile(r"```")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")


def _message_text(message: object) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _looks_like_api_question(text: str) -> bool:
    if not text:
        return False
    if _CODE_FENCE_RE.search(text):
        return True
    if any(pattern.search(text) for pattern in _API_QUESTION_PATTERNS):
        return True
    if _CAMEL_CASE_RE.search(text) or _PASCAL_CASE_RE.search(text):
        return True
    # Inline code that contains a parenthesis or dot looks like an API reference,
    # not just a filename — `foo.bar`, `Class.method`, `fn(arg)`.
    for match in _INLINE_CODE_RE.findall(text):
        inner = match.strip("`")
        if any(ch in inner for ch in "().") and not inner.startswith("/"):
            return True
    return False


def _recent_doc_search_called(messages: list, lookback: int = 2) -> bool:
    """Return True if a doc-search tool was called in the last `lookback` assistant turns."""
    assistant_turns_seen = 0
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        assistant_turns_seen += 1
        for tool_call in getattr(message, "tool_calls", []) or []:
            if tool_call.get("name") in DOC_SEARCH_TOOLS:
                return True
        if assistant_turns_seen >= lookback:
            break
    return False


def _latest_user_text(messages: list) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return _message_text(message)
    return ""


class ForceDocSearchMiddleware(AgentMiddleware):
    """Force a doc-search tool call when the user asks about specific API surfaces."""

    def __init__(self, forced_tool: str = FORCED_TOOL_NAME, lookback_turns: int = 2):
        """Initialize with the doc-search tool to force and the assistant-turn lookback."""
        super().__init__()
        self.forced_tool = forced_tool
        self.lookback_turns = lookback_turns

    def _should_force(self, request: ModelRequest) -> bool:
        # Only force on the first hop of an assistant turn — once the model has
        # already produced tool calls in this turn, the loop is reacting to tool
        # output and forcing again would clobber synthesis.
        messages = request.messages or []
        if messages and isinstance(messages[-1], AIMessage):
            return False
        user_text = _latest_user_text(messages)
        if not _looks_like_api_question(user_text):
            return False
        if _recent_doc_search_called(messages, lookback=self.lookback_turns):
            return False
        available = {
            tool.name if hasattr(tool, "name") else tool.get("name")
            for tool in (request.tools or [])
        }
        return self.forced_tool in available

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Override `tool_choice` to force a doc search when the user message demands grounding."""
        if self._should_force(request):
            logger.info(
                "Forcing tool_choice=%s for API-shaped follow-up", self.forced_tool
            )
            request = request.override(
                tool_choice={"type": "tool", "name": self.forced_tool}
            )
        return await handler(request)


__all__ = ["ForceDocSearchMiddleware"]
