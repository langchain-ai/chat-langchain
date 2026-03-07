"""Tests for the maximum search rounds cap in the docs agent prompt.

Root cause: prompt had no limit on search rounds; agent made 12+ tool calls
(9 SearchDocsByLangChain + 3 support article searches) in a single turn,
causing 6+ minute latency and ~37K tokens wasted on failed searches.

Supporting traces: 019cc703-8fb9, 019cc700-7443
"""

import re

import pytest

from src.prompts.docs_agent_prompt import docs_agent_prompt


class TestMaxSearchRoundsCap:
    """Verify the prompt contains a hard cap on the number of search rounds."""

    def test_prompt_contains_max_rounds_instruction(self):
        """The prompt must contain an explicit cap on the number of search rounds.

        Acceptable forms include:
        - "3 rounds" (the target fix)
        - "maximum" combined with "search" and "rounds"
        - "max" combined with "rounds"
        """
        prompt_lower = docs_agent_prompt.lower()

        has_three_rounds = "3 rounds" in prompt_lower
        has_maximum_rounds = (
            "maximum" in prompt_lower
            and "round" in prompt_lower
        )
        has_max_rounds = (
            "max" in prompt_lower
            and "round" in prompt_lower
            and "search" in prompt_lower
        )

        assert has_three_rounds or has_maximum_rounds or has_max_rounds, (
            "The prompt must contain a hard cap on search rounds such as "
            "'Maximum 3 search rounds total' or similar. "
            "Without this cap, the agent can make unlimited tool calls "
            "(see trace 019cc703-8fb9: 12+ calls, 6+ min latency, ~37K tokens wasted)."
        )

    def test_prompt_contains_specific_round_limit(self):
        """The prompt should specifically mention '3 rounds' as the cap."""
        assert "3 rounds" in docs_agent_prompt, (
            "The prompt should contain '3 rounds' to give the agent a concrete "
            "and actionable stopping point for follow-up searches."
        )

    def test_prompt_does_not_have_unlimited_search_directive(self):
        """The prompt must NOT contain 'Continue until you have comprehensive information'
        without a corresponding round cap.

        That phrase creates an open-ended loop. If it still appears in the prompt
        the round-limit fix has not been applied.
        """
        assert "Continue until you have comprehensive information" not in docs_agent_prompt, (
            "The unlimited search directive 'Continue until you have comprehensive "
            "information' must be replaced with a hard round cap. "
            "This open-ended instruction caused the agent to make 12+ tool calls "
            "in trace 019cc703-8fb9 before synthesizing a response."
        )

    def test_prompt_instructs_synthesize_after_max_rounds(self):
        """After reaching the max rounds the prompt should tell the agent to synthesize."""
        prompt_lower = docs_agent_prompt.lower()

        has_synthesize_after_rounds = (
            "synthesize" in prompt_lower
            and "round" in prompt_lower
        )

        assert has_synthesize_after_rounds, (
            "The prompt must instruct the agent to synthesize with available "
            "information after reaching the maximum number of search rounds, "
            "e.g. 'after 3 rounds, synthesize with what you have'."
        )

    def test_max_rounds_cap_is_in_follow_up_search_section(self):
        """The round cap should appear near the follow-up searches section."""
        # Find the follow-up searches section
        follow_up_idx = docs_agent_prompt.find("Follow-up searches")
        assert follow_up_idx != -1, "Could not find 'Follow-up searches' section in prompt"

        # The cap should appear within a reasonable distance of the follow-up section
        # (within the next 500 characters)
        section_text = docs_agent_prompt[follow_up_idx : follow_up_idx + 500]
        section_lower = section_text.lower()

        has_cap = "3 rounds" in section_lower or (
            "maximum" in section_lower and "round" in section_lower
        )

        assert has_cap, (
            "The maximum search rounds cap must appear in or near the "
            "'Follow-up searches' section of the Research Workflow, "
            "not somewhere else in the prompt where it may be ignored."
        )
