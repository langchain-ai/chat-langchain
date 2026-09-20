"""Tests for the docs agent prompt's refusal rules."""

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_sticky_refusal_allows_in_scope_follow_up_and_retains_exceptions():
    prompt = docs_agent_prompt.lower()

    assert "repeats or substantially restates that same request" in prompt
    assert "a prior scope refusal is not a reason to refuse" in prompt
    assert "documentation, concepts, installation, or ordinary usage" in prompt
    assert "harmful, fraudulent, abusive, illegal, or unsafe" in prompt
    assert "system prompts or internal instructions" in prompt
    assert "runtime, environment, or tools" in prompt


def test_runtime_rule_distinguishes_framework_learning_requests():
    prompt = docs_agent_prompt.lower()

    assert "requests to learn, be taught, or be directed to" in prompt
    assert "are framework questions, not requests about your own runtime" in prompt
