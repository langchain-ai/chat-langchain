"""Tests for public GitHub source tool validation and output bounds."""

import pytest
from pydantic import ValidationError

from src.tools.github_source_tools import (
    MAX_OUTPUT_CHARS,
    GitHubSourceInput,
    _cap_text,
)


def test_rejects_non_allowed_repository():
    with pytest.raises(ValidationError, match="repo must be one of"):
        GitHubSourceInput(operation="read", repo="someone/private-repo", path="src/main.py")


def test_rejects_unsafe_repository_path():
    with pytest.raises(ValidationError, match="relative repository path"):
        GitHubSourceInput(operation="read", repo="langchain-ai/langchain", path="../secret.py")


def test_rejects_missing_operation_arguments():
    with pytest.raises(ValidationError, match="query is required"):
        GitHubSourceInput(operation="search")

    with pytest.raises(ValidationError, match="repo and path are required"):
        GitHubSourceInput(operation="read")


def test_caps_tool_output_size():
    result = _cap_text("x" * (MAX_OUTPUT_CHARS + 100))

    assert len(result) <= MAX_OUTPUT_CHARS
    assert result.endswith("\n...[output truncated]")
