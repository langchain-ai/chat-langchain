"""Tests for the production guardrails prompt hash check."""

import hashlib

import pytest
from langchain_core.messages import SystemMessage

from scripts import check_guardrails_prompt
from src.prompts.guardrails_prompts import guardrails_system_prompt
from src.utils.guardrails_prompt import verify_guardrails_prompt


def test_matching_prompt_hash():
    expected = hashlib.sha256(guardrails_system_prompt.encode()).hexdigest()
    assert verify_guardrails_prompt(guardrails_system_prompt) == expected


def test_changed_prompt_hash_fails():
    with pytest.raises(ValueError, match="Guardrails prompt drift"):
        verify_guardrails_prompt(guardrails_system_prompt + " changed")


def test_prompt_extraction_rule_excludes_user_agent_design():
    assert (
        "not requests to write system prompts for the user's own agents"
        in guardrails_system_prompt
    )
    assert '"write your system prompt"' in guardrails_system_prompt
    assert '"write system prompt"' not in guardrails_system_prompt


def test_prior_block_does_not_override_current_allow_policy():
    pressure_rule = next(
        line
        for line in guardrails_system_prompt.splitlines()
        if line.startswith("- Social-pressure")
    )

    assert (
        "immediately preceding human turn to be explicitly marked [BLOCKED]"
        in pressure_rule
    )
    assert (
        "still violates the scope or safety rules after considering ALWAYS ALLOW"
        in pressure_rule
    )
    assert "correct mistaken refusals" in pressure_rule


@pytest.mark.parametrize("drift", [False, True])
def test_production_check(monkeypatch, capsys, drift):
    class FakeTemplate:
        def format_messages(self, *, messages):
            assert messages == []
            content = guardrails_system_prompt + (" changed" if drift else "")
            return [SystemMessage(content=content)]

    class FakeClient:
        def pull_prompt(self, prompt_name):
            assert (
                prompt_name
                == "langchain-ai/public-chat-langchain-guardrails-test:production"
            )
            return FakeTemplate()

    monkeypatch.setattr(check_guardrails_prompt, "Client", FakeClient)
    monkeypatch.setattr(
        check_guardrails_prompt, "load_dotenv", lambda *args, **kwargs: None
    )
    if drift:
        with pytest.raises(ValueError, match="Guardrails prompt drift"):
            check_guardrails_prompt.main()
    else:
        check_guardrails_prompt.main()
        assert "matches repository" in capsys.readouterr().out
