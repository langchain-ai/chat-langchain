"""Tests for deterministic final-answer citation validation."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.citation_guard_middleware import CitationGuardMiddleware


@pytest.mark.asyncio
async def test_gate_validates_missing_urls_and_removes_invalid_links():
    """The gate validates only missing URLs and removes failed citations."""
    state = {
        "messages": [
            HumanMessage(content="Find docs"),
            ToolMessage(
                name="check_links",
                tool_call_id="1",
                content="Link Check Results: 1/1 valid\n\nValid links:\n  - https://example.com/ok",
            ),
            AIMessage(
                content=(
                    "[ok](https://example.com/ok) "
                    "[missing](https://example.com/missing#section)"
                )
            ),
        ]
    }
    check = AsyncMock(
        return_value=(
            "Link Check Results: 1/1 valid\n\nValid links:\n"
            "  - https://example.com/missing#section"
        )
    )

    with patch("src.middleware.citation_guard_middleware.check_links.coroutine", check):
        update = await CitationGuardMiddleware().aafter_agent(state, None)

    assert update is None
    check.assert_awaited_once_with(urls=["https://example.com/missing#section"])


@pytest.mark.asyncio
async def test_gate_removes_invalid_url_without_reinvoking_model():
    """Invalid URLs are removed after one deterministic tool validation."""
    state = {
        "messages": [
            HumanMessage(content="Find docs"),
            AIMessage(content="[bad](https://example.com/missing#section)"),
        ]
    }
    check = AsyncMock(
        return_value="Link Check Results: 0/1 valid\n\nInvalid links:\n"
        "  - https://example.com/missing#section: anchor not found on page"
    )

    with patch("src.middleware.citation_guard_middleware.check_links.coroutine", check):
        update = await CitationGuardMiddleware().aafter_agent(state, None)

    assert update["messages"][-1].content == "bad"
    check.assert_awaited_once_with(urls=["https://example.com/missing#section"])
