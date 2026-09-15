"""Tests for search result provenance in read-tool instructions."""

from pathlib import Path

from src.prompts.docs_agent_prompt import docs_agent_prompt

instructions = Path("instructions.md").read_text()


def test_read_tools_require_current_turn_targets():
    """Read instructions require provenance and permit skipping invalid reads."""
    for prompt in (instructions, docs_agent_prompt):
        assert "copied verbatim" in prompt
        assert "current-turn" in prompt
        assert "skip" in prompt
        assert "do not fabricate" in prompt
        assert "read_file" in prompt
        assert "guessed `.mdx` suffixes" in prompt
        assert "ALWAYS follow up by reading" not in prompt
