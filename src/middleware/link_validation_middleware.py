"""Validate final-answer links without an additional model turn."""

import asyncio
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.tools.link_check_tools import _check_urls_async

logger = logging.getLogger(__name__)
VALIDATION_BUDGET = 5.0
_URL = r"https?://(?:[^\s<>`\[\]()]+|\([^\s<>`\[\]()]*\))+"
_LINK = re.compile(
    rf'\[(?P<label>[^\]\n]*)\]\(<?(?P<markdown>{_URL})>?(?:\s+"[^"\n]*")?\)'
    rf"|<(?P<auto>{_URL})>|(?P<bare>{_URL})"
)
_LIST_ITEM = re.compile(r"^\s*(?:[-+*]|\d+[.)])\s+(.*?)\s*$")


def _url(match: re.Match) -> str:
    """Extract a link destination without prose punctuation."""
    return match["markdown"] or match["auto"] or match["bare"].rstrip(".,;:!?")


def _remove_invalid(text: str, invalid: set[str]) -> str:
    """Remove rejected destinations while keeping descriptive prose."""
    lines = []
    for line in text.splitlines(keepends=True):
        item = _LIST_ITEM.fullmatch(line.rstrip("\r\n"))
        link = _LINK.fullmatch(item[1]) if item else None
        if link and _url(link) in invalid:
            continue

        def replace(match: re.Match) -> str:
            if _url(match) not in invalid:
                return match[0]
            if match["markdown"]:
                return match["label"]
            if match["bare"]:
                return match[0][len(_url(match)) :]
            return ""

        lines.append(_LINK.sub(replace, line))
    return "".join(lines)


class LinkValidationMiddleware(AgentMiddleware):
    """Check URLs in the final model message with a fail-open budget."""

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Replace invalid final-answer links in the existing message."""
        messages = state.get("messages", [])
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        if message.tool_calls:
            return None
        content = message.content
        blocks = [content] if isinstance(content, str) else content
        texts = [
            block if isinstance(block, str) else block["text"]
            for block in blocks
            if isinstance(block, str)
            or (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )
        ]
        urls = list(
            dict.fromkeys(_url(m) for text in texts for m in _LINK.finditer(text))
        )
        if not urls:
            return None
        try:
            results = await asyncio.wait_for(
                _check_urls_async(urls, VALIDATION_BUDGET), timeout=VALIDATION_BUDGET
            )
            if any(result.operational_failure for result in results):
                return None
            invalid = {result.url for result in results if not result.valid}
        except Exception:
            logger.warning(
                "Final-answer link validation failed; preserving answer", exc_info=True
            )
            return None
        if not invalid:
            return None
        updated: str | list[str | dict]
        if isinstance(content, str):
            updated = _remove_invalid(content, invalid)
        else:
            updated = [
                _remove_invalid(block, invalid)
                if isinstance(block, str)
                else {**block, "text": _remove_invalid(block["text"], invalid)}
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
                else block
                for block in content
            ]
        return {"messages": [message.model_copy(update={"content": updated})]}
