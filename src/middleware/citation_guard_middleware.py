"""Deterministic validation for URLs in final agent answers."""

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

from src.tools.link_check_tools import check_links

_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)")


def _message_text(content: Any) -> str:
    """Return text content from a LangChain message payload."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _answer_urls(content: str) -> list[str]:
    """Extract markdown link targets from an answer."""
    return list(dict.fromkeys(_MARKDOWN_LINK_RE.findall(content)))


def _valid_urls_from_results(messages: list[Any]) -> set[str]:
    """Extract exact URLs from check_links valid-link sections."""
    valid_urls: set[str] = set()
    for message in messages:
        if getattr(message, "name", None) != "check_links":
            continue
        valid_urls.update(_valid_urls_from_text(_message_text(getattr(message, "content", ""))))
    return valid_urls


def _valid_urls_from_text(content: str) -> set[str]:
    """Extract exact valid URLs from one check_links result."""
    valid_urls: set[str] = set()
    in_valid_section = False
    for line in content.splitlines():
        if line.strip() == "Valid links:":
            in_valid_section = True
            continue
        if in_valid_section and line.strip().startswith("- "):
            valid_urls.add(line.strip()[2:].split(" (→ ", 1)[0])
        elif in_valid_section and line.strip() and not line.startswith("  - "):
            in_valid_section = False
    return valid_urls


def _remove_unvalidated_links(content: str, invalid_urls: set[str]) -> str:
    """Remove markdown links whose exact targets are not valid."""
    def replace(match: re.Match[str]) -> str:
        url = match.group(1)
        return match.group(0) if url not in invalid_urls else match.group(0).split("](", 1)[0][1:]

    return _MARKDOWN_LINK_RE.sub(replace, content)


class CitationGuardMiddleware(AgentMiddleware):
    """Validate every final-answer markdown link without calling the model."""

    async def aafter_agent(
        self, state: dict[str, Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Validate and remove unverified links from the final answer."""
        messages = list(state.get("messages", []))
        final_message = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if final_message is None or not isinstance(final_message.content, str):
            return None

        answer_urls = _answer_urls(final_message.content)
        if not answer_urls:
            return None

        current_turn_start = max(
            (index for index, message in enumerate(messages) if getattr(message, "type", None) == "human"),
            default=0,
        )
        turn_messages = messages[current_turn_start:]
        checked_urls = _valid_urls_from_results(turn_messages)
        missing_urls = [url for url in answer_urls if url not in checked_urls]
        if missing_urls:
            check_result = await check_links.coroutine(urls=missing_urls)
            checked_urls.update(_valid_urls_from_text(check_result))

        invalid_urls = set(answer_urls) - checked_urls
        if not invalid_urls:
            return None

        updated_message = final_message.model_copy(
            update={"content": _remove_unvalidated_links(final_message.content, invalid_urls)}
        )
        messages[messages.index(final_message)] = updated_message
        return {"messages": messages}


__all__ = ["CitationGuardMiddleware"]
