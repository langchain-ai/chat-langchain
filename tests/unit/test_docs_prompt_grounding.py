"""Tests for consistent no-results grounding instructions."""

from pathlib import Path

import pytest

from src.prompts.docs_agent_prompt import docs_agent_prompt


@pytest.mark.parametrize(
    "prompt", [docs_agent_prompt, Path("instructions.md").read_text()]
)
def test_unsuccessful_search_guidance_is_honest_and_non_speculative(prompt):
    assert (
        'state: "I could not find relevant documentation to answer this question."'
        in prompt
    )
    assert (
        "Unsuccessful searches do not establish that documentation does not exist or does not cover the topic"
        in prompt
    )
    assert "do not infer absence or speculate that documentation moved" in prompt
    assert "Do not recommend legacy sites or invent replacement destinations" in prompt
    assert "NEVER include links to python.langchain.com or js.langchain.com" in prompt
    assert 'Only include URLs that `check_links` returns under "Valid links"' in prompt
    assert (
        "Offer a closest relevant page only when retrieved evidence supports its relevance and `check_links` validates its URL"
        in prompt
    )
    assert "STOP and use docs.langchain.com instead" not in prompt
