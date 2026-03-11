"""Unit tests verifying the docs_agent_prompt explicitly and prominently bans stale URLs.

Root cause: The agent generates python.langchain.com / js.langchain.com links from
memory even when SearchDocsByLangChain returns correct docs.langchain.com URLs. The
prompt instruction was present but buried in "Important Customer Service Rules" near
the bottom — not prominent enough for the model to consistently follow.

These tests verify that the prompt:
1. Explicitly names the banned domains (python.langchain.com, js.langchain.com)
2. Contains a URL domain rule that bans both stale domains
3. Mentions the ban prominently — specifically in the Formatting Validation Checklist,
   so it appears both in the main rules AND in the pre-send checklist.
"""

import pytest

from src.prompts.docs_agent_prompt import docs_agent_prompt

STALE_DOMAINS = ["python.langchain.com", "js.langchain.com"]


class TestStaleUrlsAreBanned:
    """The prompt must name and ban both stale documentation domains."""

    @pytest.mark.parametrize("domain", STALE_DOMAINS)
    def test_prompt_names_banned_domain(self, domain):
        """Each stale domain must be explicitly named somewhere in the prompt."""
        assert domain in docs_agent_prompt, (
            f"The prompt does not mention '{domain}'. "
            "The stale-URL ban must explicitly name the forbidden domain so the model "
            "recognises it and does not generate links from training-data memory."
        )

    @pytest.mark.parametrize("domain", STALE_DOMAINS)
    def test_banned_domain_appears_multiple_times(self, domain):
        """Each stale domain should appear at least twice (rule + checklist)."""
        count = docs_agent_prompt.count(domain)
        assert count >= 2, (
            f"'{domain}' appears only {count} time(s) in the prompt. "
            "It should appear at least twice — once in the URL rules section and "
            "once in the Formatting Validation Checklist — to be prominent enough "
            "for the model to reliably follow."
        )

    def test_checklist_bans_stale_domains(self):
        """The Formatting Validation Checklist must reference the stale-URL ban."""
        # Find the checklist section
        checklist_marker = "Formatting Validation Checklist"
        assert checklist_marker in docs_agent_prompt, (
            "Could not find 'Formatting Validation Checklist' section in prompt."
        )

        checklist_start = docs_agent_prompt.index(checklist_marker)
        # The checklist runs to the end of the section; grab a generous chunk
        checklist_text = docs_agent_prompt[checklist_start : checklist_start + 2000]

        for domain in STALE_DOMAINS:
            assert domain in checklist_text, (
                f"'{domain}' is not mentioned in the Formatting Validation Checklist. "
                "Adding it there ensures the model checks for stale URLs as a final "
                "step before every response, not just as a buried background rule."
            )

    def test_stale_url_rule_appears_before_halfway_point(self):
        """The stale-URL prohibition must appear in the first half of the prompt.

        If the rule only appears near the bottom, the model's attention fades and it
        reverts to training-data memory for URL generation.
        """
        midpoint = len(docs_agent_prompt) // 2
        first_half = docs_agent_prompt[:midpoint]

        # At least one of the stale domains must appear before the midpoint
        found_early = any(domain in first_half for domain in STALE_DOMAINS)
        assert found_early, (
            "Neither 'python.langchain.com' nor 'js.langchain.com' appears in the "
            "first half of the prompt. The stale-URL ban must be prominent (appearing "
            "early or in repeated sections) so the model follows it consistently."
        )
