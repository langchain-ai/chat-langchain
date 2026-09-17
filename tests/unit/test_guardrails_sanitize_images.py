"""Tests for sanitizing empty base64 image blocks before model dispatch."""

import os

os.environ["USE_LOCAL_PROMPTS"] = "1"

from langchain_core.messages import HumanMessage

from src.middleware.guardrails_middleware import _sanitize_image_blocks


def test_sanitize_replaces_empty_base64_image_block():
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,"}},
        ]
    )

    _sanitize_image_blocks([msg])

    assert msg.content[0] == {"type": "text", "text": "What's in this image?"}
    assert msg.content[1] == {
        "type": "text",
        "text": "[Invalid image upload — please re-attach the image]",
    }


def test_sanitize_preserves_valid_image_block():
    valid_url = "data:image/png;base64,iVBORw0KGgoAAAA"
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Describe."},
            {"type": "image_url", "image_url": {"url": valid_url}},
        ]
    )

    _sanitize_image_blocks([msg])

    assert msg.content[1] == {"type": "image_url", "image_url": {"url": valid_url}}


def test_sanitize_skips_string_content():
    msg = HumanMessage(content="data:image/png;base64,")

    _sanitize_image_blocks([msg])

    assert msg.content == "data:image/png;base64,"
