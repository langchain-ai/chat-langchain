"""Validate language consistency in docs-agent responses."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Sequence

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

_LANGUAGE_PATH_RE = re.compile(
    r"https?://docs\.langchain\.com/oss/(python|javascript)(?:/|$)"
)
_CODE_FENCE_RE = re.compile(r"(?ms)^```([\w+-]*)\s*\n(.*?)^```")
_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+.*$")
_PYTHON_BANNED_PATTERNS = (
    re.compile(r"\baddEdge\b"),
    re.compile(r"\baddNode\b"),
    re.compile(r"\baddConditionalEdges\b"),
    re.compile(r"\bnew\s+StateGraph\b"),
    re.compile(r"\bconst\b"),
)
_REGENERATION_INSTRUCTION = (
    "Your previous response failed a language consistency check. Regenerate the complete "
    "answer now: use Python API spellings and /oss/python/ citations for Python answers, "
    "or JavaScript/TypeScript API spellings and /oss/javascript/ citations for JavaScript "
    "answers. Keep each fenced code block and its citations in the same language."
)


def infer_answer_language(messages: Sequence[BaseMessage]) -> str:
    """Infer the requested answer language, defaulting to Python."""
    for message in reversed(messages):
        text = _message_text(message)
        if re.search(r"\b(?:javascript|typescript|js|ts)\b", text, re.IGNORECASE):
            return "javascript"
        if re.search(r"\b(?:python|py)\b", text, re.IGNORECASE):
            return "python"
    return "python"


def validate_response_language(content: str, default_language: str) -> bool:
    """Return whether code and documentation citations use matching languages."""
    fences = list(_CODE_FENCE_RE.finditer(content))
    if not fences:
        return _citations_match_language(content, default_language)

    for index, fence in enumerate(fences):
        fence_language = _normalize_language(fence.group(1)) or default_language
        code = fence.group(2)
        if fence_language == "python" and any(
            pattern.search(code) for pattern in _PYTHON_BANNED_PATTERNS
        ):
            return False

        supporting_text = _supporting_text(content, fences, index)
        if not _citations_match_language(supporting_text, fence_language):
            return False
    return True


class ResponseLanguageValidationError(ValueError):
    """Raised when regeneration still returns a language-mismatched response."""


class ResponseLanguageMiddleware(AgentMiddleware):
    """Regenerate one response when code and citations use the wrong language."""

    def _validate(self, response: ModelResponse, request: ModelRequest) -> bool:
        message = _latest_ai_message(response.result)
        if message is None or not isinstance(message.content, str):
            return True
        language = infer_answer_language(request.messages)
        return validate_response_language(message.content, language)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Regenerate one invalid synchronous model response."""
        response = handler(request)
        if self._validate(response, request):
            return response
        regenerated = handler(_regeneration_request(request))
        if not self._validate(regenerated, request):
            raise ResponseLanguageValidationError(
                "Model response failed language consistency validation after regeneration"
            )
        return regenerated

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Regenerate one invalid asynchronous model response."""
        response = await handler(request)
        if self._validate(response, request):
            return response
        regenerated = await handler(_regeneration_request(request))
        if not self._validate(regenerated, request):
            raise ResponseLanguageValidationError(
                "Model response failed language consistency validation after regeneration"
            )
        return regenerated


def _message_text(message: BaseMessage) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def _latest_ai_message(messages: Sequence[BaseMessage]) -> AIMessage | None:
    return next(
        (message for message in reversed(messages) if isinstance(message, AIMessage)),
        None,
    )


def _normalize_language(language: str) -> str | None:
    normalized = language.lower()
    if normalized in {"py", "python"}:
        return "python"
    if normalized in {"js", "javascript", "ts", "typescript"}:
        return "javascript"
    return None


def _citations_match_language(text: str, language: str) -> bool:
    return all(
        path_language == language for path_language in _LANGUAGE_PATH_RE.findall(text)
    )


def _section_for_position(content: str, position: int) -> str:
    headings = list(_HEADING_RE.finditer(content))
    previous_heading = next(
        (heading for heading in reversed(headings) if heading.start() < position), None
    )
    next_heading = next(
        (heading for heading in headings if heading.start() > position), None
    )
    start = previous_heading.start() if previous_heading else 0
    end = next_heading.start() if next_heading else len(content)
    return content[start:end]


def _supporting_text(content: str, fences: list[re.Match[str]], index: int) -> str:
    if len(fences) == 1:
        return content
    section = _section_for_position(content, fences[index].start())
    if section != content:
        return section
    start = fences[index - 1].end() if index else 0
    end = fences[index + 1].start() if index + 1 < len(fences) else len(content)
    return content[start:end]


def _regeneration_request(request: ModelRequest) -> ModelRequest:
    current = request.system_message.text if request.system_message else ""
    system_message = f"{current}\n\n{_REGENERATION_INSTRUCTION}".strip()
    return request.override(system_message=SystemMessage(content=system_message))


__all__ = [
    "ResponseLanguageMiddleware",
    "ResponseLanguageValidationError",
    "infer_answer_language",
    "validate_response_language",
]
