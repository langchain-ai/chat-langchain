from pathlib import Path

from src.prompts.docs_agent_prompt import docs_agent_prompt

INSTRUCTIONS = Path(__file__).parents[2].joinpath("instructions.md").read_text()


def test_new_in_scope_follow_up_requires_retrieval_after_refusal():
    """A new in-scope question must not inherit a previous refusal."""
    for prompt in (docs_agent_prompt, INSTRUCTIONS):
        assert "Refusals apply only to the refused request" in prompt
        assert "Any NEW question within the LangChain, LangGraph, LangSmith, Fleet, or DeepAgents scope" in prompt
        assert "never reuse a prior refusal as the answer to a different question" in prompt
        assert "Run at least one documentation search before declining for scope" in prompt


def test_beginner_langgraph_teaching_request_is_not_prompt_extraction():
    """Beginner-friendly LangGraph teaching requests remain ordinary docs questions."""
    for prompt in (docs_agent_prompt, INSTRUCTIONS):
        assert "This rule applies only when the user asks for your own prompt, instructions, tool list, or runtime configuration" in prompt
        assert "explain like I'm a beginner/child" in prompt
        assert "teach me from scratch" in prompt
        assert "show me the documentation" in prompt
        assert "must be researched and answered" in prompt
