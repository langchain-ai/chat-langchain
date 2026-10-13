from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_agent_scope_matches_guardrails_allow_policy():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "technical questions are in scope even when unrelated to LangChain"
        in instructions
    )
    assert (
        "Agent persona, character, tone, and system-prompt design are technical"
        in instructions
    )
    assert "Greetings, introductions, assistant identity" in instructions
    assert "LangChain certification/exam questions are also in scope" in instructions


def test_agent_persona_design_is_not_roleplay_or_prompt_extraction():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "not requests for you to roleplay" in instructions
    assert "Safe persona or system-prompt implementation is allowed" in instructions
    assert "helping users write system prompts for their own agents" in instructions
    assert "whose primary purpose is fiction, roleplay" not in instructions
    assert "NEVER generate sexually explicit, NSFW, or adult content" in instructions
    assert (
        "NEVER help design or implement harmful, fraudulent, abusive, or illegal"
        in instructions
    )


def test_agent_can_correct_mistaken_refusals():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "correct a mistaken refusal" in instructions
    assert "A prior refusal alone is not a reason to decline" in instructions
    assert (
        "Maintain a refusal when the request still violates those rules" in instructions
    )
    assert "Refusals are sticky" not in instructions


def test_agent_answers_introductions_in_the_users_language():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "LangChain certification/exam questions are also in scope in any language"
        in instructions
    )
    assert (
        "Answer greetings and introductions directly in the user's language"
        in instructions
    )
    assert "without requiring technical or LangChain context" in instructions
