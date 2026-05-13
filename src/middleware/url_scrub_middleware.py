# Defense-in-depth middleware that strips fabricated docs.langchain.com/docs/...
# URLs from the final assistant message.
#
# Background: the docs site no longer hosts content under the legacy `/docs/...`
# path prefix (the canonical layout is `/oss/python/...`, `/oss/javascript/...`,
# `/langsmith/...`, `/langgraph-platform/...`, `/labs/...`). The agent
# occasionally fabricates `https://docs.langchain.com/docs/...` links from its
# training-data memory, which produces 404s for users. The prompt forbids these
# URLs, but this middleware is a belt-and-suspenders guard so an unvalidated
# fabricated URL never ships in a response.
import logging
import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# Match Markdown links whose URL is `https://docs.langchain.com/docs/...`.
# We only scrub the legacy `/docs/` prefix — real `/oss/...`, `/langsmith/...`,
# etc. URLs on the same host are preserved.
_FABRICATED_MD_LINK_RE = re.compile(
    r"\[([^\]]+)\]\(https://docs\.langchain\.com/docs/[^)]*\)"
)

# Also strip bare (non-Markdown) fabricated URLs that may slip through.
_FABRICATED_BARE_URL_RE = re.compile(
    r"https://docs\.langchain\.com/docs/\S+"
)

_REPLACEMENT_MARKER = "[FABRICATED LINK REMOVED]"


def scrub_fabricated_docs_urls(text: str) -> tuple[str, int]:
    """Remove fabricated docs.langchain.com/docs/... URLs from text.

    Returns the scrubbed text and the count of links replaced. Markdown links
    are replaced with a `[FABRICATED LINK REMOVED]` marker so the surrounding
    bullet/structure stays intact; bare URLs are replaced with the same marker.
    """
    replaced = 0

    def _replace_md(_match: re.Match[str]) -> str:
        nonlocal replaced
        replaced += 1
        return _REPLACEMENT_MARKER

    scrubbed = _FABRICATED_MD_LINK_RE.sub(_replace_md, text)

    def _replace_bare(_match: re.Match[str]) -> str:
        nonlocal replaced
        replaced += 1
        return _REPLACEMENT_MARKER

    scrubbed = _FABRICATED_BARE_URL_RE.sub(_replace_bare, scrubbed)
    return scrubbed, replaced


def _scrub_message_content(content: Any) -> tuple[Any, int]:
    """Scrub fabricated URLs from a message `content` value.

    Supports plain string content and the list-of-content-blocks shape used by
    some providers. Returns the (possibly mutated) content and a replacement
    count.
    """
    if isinstance(content, str):
        return scrub_fabricated_docs_urls(content)

    if isinstance(content, list):
        total = 0
        new_blocks: list[Any] = []
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                new_text, count = scrub_fabricated_docs_urls(block["text"])
                if count:
                    block = {**block, "text": new_text}
                    total += count
            new_blocks.append(block)
        return new_blocks, total

    return content, 0


class UrlScrubMiddleware(AgentMiddleware[AgentState]):
    """Strip fabricated `docs.langchain.com/docs/...` URLs from model output.

    Runs in the `after_model` hook so it sees the final assistant message before
    it is returned to the user. If any fabricated URLs are found, they are
    replaced with a `[FABRICATED LINK REMOVED]` marker and the replacement
    count is logged for observability.
    """

    def _scrub_state(self, state: AgentState) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None

        last = messages[-1]
        if not isinstance(last, AIMessage):
            return None

        new_content, replaced = _scrub_message_content(last.content)
        if not replaced:
            return None

        logger.warning(
            "url_scrub_middleware removed %d fabricated docs.langchain.com/docs/* "
            "link(s) from final response",
            replaced,
        )

        # Preserve message id so LangGraph treats this as an update to the same
        # message rather than appending a new one.
        scrubbed = AIMessage(
            content=new_content,
            id=getattr(last, "id", None),
            additional_kwargs=getattr(last, "additional_kwargs", {}) or {},
            response_metadata=getattr(last, "response_metadata", {}) or {},
            tool_calls=getattr(last, "tool_calls", []) or [],
        )
        return {"messages": [scrubbed]}

    def after_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return self._scrub_state(state)

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return self._scrub_state(state)
