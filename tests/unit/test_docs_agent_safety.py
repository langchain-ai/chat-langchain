"""Tests for docs-agent host-action restrictions and tool visibility."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ.setdefault("ANTHROPIC_API_KEY", "test")

from deepagents import GeneralPurposeSubagentProfile, HarnessProfile
from deepagents.profiles.harness.harness_profiles import _HARNESS_PROFILES

from agent import docs_agent_tools
from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_docs_prompt_rejects_downloadable_file_requests_while_answering():
    prompt = docs_agent_prompt.lower()

    assert "cannot" in prompt
    assert "downloadable file" in prompt
    assert "informational component" in prompt
    assert "delivery or execution limitation" in prompt


def test_docs_agent_profile_exposes_only_authored_and_mcp_tools():
    profile = _HARNESS_PROFILES["google_genai:gemini-3.5-flash-lite"]
    expected_excluded_tools = {
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "glob",
        "grep",
        "execute",
    }

    assert isinstance(profile, HarnessProfile)
    assert profile.excluded_tools == expected_excluded_tools
    assert profile.general_purpose_subagent == GeneralPurposeSubagentProfile(enabled=False)
    assert {tool.name for tool in docs_agent_tools} == {
        "search_support_articles",
        "get_support_article_content",
        "fetch_langchain_pricing",
        "check_links",
    }
