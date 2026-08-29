from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.middleware.link_validation_middleware import LinkValidationMiddleware


@pytest.mark.asyncio
async def test_unvalidated_url_is_checked_and_removed():
    message = AIMessage(content="See [the guide](https://docs.langchain.com/missing).")
    state = {"messages": [message]}
    tool_result = "Link Check Results: 0/1 valid\n\nInvalid links:\n  - https://docs.langchain.com/missing: 404"

    check = AsyncMock(return_value=tool_result)
    with patch(
        "src.middleware.link_validation_middleware.check_links",
        new=SimpleNamespace(ainvoke=check),
    ):
        update = await LinkValidationMiddleware().aafter_model(
            state, runtime=SimpleNamespace()
        )

    check.assert_awaited_once_with({"urls": ["https://docs.langchain.com/missing"]})
    assert update["messages"][0].content == "See the guide."


@pytest.mark.asyncio
async def test_validated_url_passes_through_untouched():
    url = "https://docs.langchain.com/concepts"
    message = AIMessage(content=f"See [the guide]({url}).")
    tool_message = ToolMessage(
        name="check_links",
        tool_call_id="call-1",
        content=f"Link Check Results: 1/1 valid\n\nValid links:\n  - {url}",
    )

    check = AsyncMock()
    with patch(
        "src.middleware.link_validation_middleware.check_links",
        new=SimpleNamespace(ainvoke=check),
    ):
        update = await LinkValidationMiddleware().aafter_model(
            {"messages": [tool_message, message]}, runtime=SimpleNamespace()
        )

    check.assert_not_awaited()
    assert update is None


@pytest.mark.asyncio
async def test_validated_base_allows_anchor_url():
    base_url = "https://docs.langchain.com/concepts"
    anchor_url = f"{base_url}#streaming"
    message = AIMessage(content=f"Read [this section]({anchor_url}).")
    tool_message = ToolMessage(
        name="check_links",
        tool_call_id="call-1",
        content=f"Link Check Results: 1/1 valid\n\nValid links:\n  - {base_url}",
    )

    check = AsyncMock()
    with patch(
        "src.middleware.link_validation_middleware.check_links",
        new=SimpleNamespace(ainvoke=check),
    ):
        update = await LinkValidationMiddleware().aafter_model(
            {"messages": [tool_message, message]}, runtime=SimpleNamespace()
        )

    check.assert_not_awaited()
    assert update is None


@pytest.mark.asyncio
async def test_content_block_list_message_rewrites_invalid_url():
    url = "https://support.langchain.com/missing"
    message = AIMessage(
        content=[
            {"type": "text", "text": "Read "},
            {"type": "text", "text": f"[the article]({url})"},
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/image"},
            },
        ]
    )
    tool_result = "Link Check Results: 0/1 valid\n\nInvalid links:\n  - " + url

    check = AsyncMock(return_value=tool_result)
    with patch(
        "src.middleware.link_validation_middleware.check_links",
        new=SimpleNamespace(ainvoke=check),
    ):
        update = await LinkValidationMiddleware().aafter_model(
            {"messages": [message]}, runtime=SimpleNamespace()
        )

    check.assert_awaited_once_with({"urls": [url]})
    assert update["messages"][0].content == [
        {"type": "text", "text": "Read "},
        {"type": "text", "text": "the article"},
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/image"},
        },
    ]
