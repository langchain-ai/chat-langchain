"""Reject undocumented or deprecated package references in docs answers."""

from __future__ import annotations

import re
from typing import Awaitable, Callable, NamedTuple

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

DOCS_TOOL_SUFFIXES = (
    "search_docs_by_lang_chain",
    "query_docs_filesystem_docs_by_lang_chain",
)
WARNING_MARKERS = (
    "deprecated",
    "no longer maintained",
    "unmaintained",
    "do not use",
)
IMPORT_FROM_PATTERN = re.compile(
    r"^\s*from\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s+import\s+([A-Za-z_]\w*)",
    re.MULTILINE,
)
IMPORT_PATTERN = re.compile(
    r"^\s*import\s+([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*(?:\s*,\s*[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)*)",
    re.MULTILINE,
)
INSTALL_PATTERN = re.compile(r"\b(?:pip install|uv add)\s+([^\n;]+)")
FENCED_CODE_PATTERN = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)


class GroundingTarget(NamedTuple):
    """A package or import reference found in a draft."""

    kind: str
    path: str
    symbol: str | None = None


class DocsGroundingError(ValueError):
    """Raised when a draft contains an unsupported documentation reference."""


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


def _current_turn_tool_results(messages: list[object]) -> list[str]:
    start = 0
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            start = index
            break

    results = []
    for message in messages[start:]:
        if not isinstance(message, ToolMessage):
            continue
        name = message.name or ""
        if any(name.endswith(suffix) for suffix in DOCS_TOOL_SUFFIXES):
            results.append(_message_text(message.content))
    return results


def _draft_code(response: ModelResponse | AIMessage) -> str:
    messages = response.result if isinstance(response, ModelResponse) else [response]
    return "\n".join(
        match.group(1) for message in messages for match in FENCED_CODE_PATTERN.finditer(
            _message_text(message.content)
        )
    )


def _draft_targets(response: ModelResponse | AIMessage) -> list[GroundingTarget]:
    code = _draft_code(response)
    targets: list[GroundingTarget] = []
    for match in IMPORT_FROM_PATTERN.finditer(code):
        targets.append(GroundingTarget("from", match.group(1), match.group(2)))
    for match in IMPORT_PATTERN.finditer(code):
        for path in match.group(1).split(","):
            targets.append(GroundingTarget("import", path.strip()))
    for match in INSTALL_PATTERN.finditer(code):
        for token in match.group(1).split():
            token = token.strip("`'\"(),")
            if token and not token.startswith("-"):
                targets.append(GroundingTarget("install", token))
    return targets


def _target_matches(target: GroundingTarget, text: str) -> list[tuple[int, int]]:
    if target.kind == "from":
        pattern = re.compile(
            rf"\bfrom\s+{re.escape(target.path)}\s+import\s+{re.escape(target.symbol or '')}\b"
        )
    elif target.kind == "import":
        pattern = re.compile(rf"\bimport\s+{re.escape(target.path)}\b")
    else:
        pattern = re.compile(rf"(?<![A-Za-z0-9_.-]){re.escape(target.path)}(?![A-Za-z0-9_.-])")
    return [match.span() for match in pattern.finditer(text)]


def _is_warning_context(text: str, span: tuple[int, int]) -> bool:
    start = max(text.rfind(".", 0, span[0]), text.rfind("\n", 0, span[0])) + 1
    end_candidates = [
        position for position in (text.find(".", span[1]), text.find("\n", span[1])) if position != -1
    ]
    end = min(end_candidates, default=len(text))
    context = text[start:end].lower()
    return any(marker in context for marker in WARNING_MARKERS)


def _target_status(target: GroundingTarget, docs: list[str]) -> str:
    matches = [span for text in docs for span in _target_matches(target, text)]
    if not matches:
        return "missing"
    for text in docs:
        for span in _target_matches(target, text):
            if not _is_warning_context(text, span):
                return "supported"
    return "deprecated"


def _invalid_targets(response: ModelResponse | AIMessage, docs: list[str]) -> list[GroundingTarget]:
    return [target for target in _draft_targets(response) if _target_status(target, docs) != "supported"]


def _retry_request(request: ModelRequest, targets: list[GroundingTarget]) -> ModelRequest:
    references = ", ".join(
        f"{target.path}.{target.symbol}" if target.symbol else target.path for target in targets
    )
    instruction = (
        "The previous draft used unsupported or deprecated technical references: "
        f"{references}. Rewrite the answer using only package names, install targets, "
        "and complete imports explicitly present in the documentation read this turn. "
        "If the documentation names no replacement, say that no documented replacement was found."
    )
    system_text = request.system_message.text if request.system_message else ""
    return request.override(
        system_message=SystemMessage(content=f"{system_text}\n\n{instruction}"),
    )


class DocsGroundingMiddleware(AgentMiddleware):
    """Retry drafts that cite undocumented or deprecated packages and imports."""

    def _validate(self, request: ModelRequest, response: ModelResponse | AIMessage) -> list[GroundingTarget]:
        return _invalid_targets(response, _current_turn_tool_results(request.messages))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Validate a synchronous model response and retry it once."""
        response = handler(request)
        invalid = self._validate(request, response)
        if not invalid:
            return response
        retry_response = handler(_retry_request(request, invalid))
        remaining = self._validate(request, retry_response)
        if remaining:
            raise DocsGroundingError("Draft contains references not supported by retrieved documentation")
        return retry_response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Validate an asynchronous model response and retry it once."""
        response = await handler(request)
        invalid = self._validate(request, response)
        if not invalid:
            return response
        retry_response = await handler(_retry_request(request, invalid))
        remaining = self._validate(request, retry_response)
        if remaining:
            raise DocsGroundingError("Draft contains references not supported by retrieved documentation")
        return retry_response


__all__ = ["DocsGroundingError", "DocsGroundingMiddleware"]
