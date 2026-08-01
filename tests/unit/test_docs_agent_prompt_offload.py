"""Tests for offload-handling guidance in the docs agent prompt."""

from pathlib import Path

from src.prompts.docs_agent_prompt import docs_agent_prompt

_INSTRUCTIONS = Path(__file__).resolve().parents[2] / "instructions.md"


def test_prompt_explains_offloaded_tool_results():
    """Prompt must tell the agent to dereference offloaded tool results."""
    assert "Tool result too large" in docs_agent_prompt
    assert "/large_tool_results/<id>" in docs_agent_prompt
    assert "query_docs_filesystem_docs_by_lang_chain" in docs_agent_prompt


def test_instructions_mirror_offload_guidance():
    """The runtime instructions must carry the same offload guidance."""
    instructions = _INSTRUCTIONS.read_text()

    assert "Tool result too large" in instructions
    assert "/large_tool_results/<id>" in instructions
