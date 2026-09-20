from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]
PROMPT_PATH = REPOSITORY_ROOT / "src/prompts/docs_agent_prompt.py"
INSTRUCTIONS_PATH = REPOSITORY_ROOT / "instructions.md"


def _refusal_section(text: str) -> str:
    start = text.index("**Refusals")
    end = text.index("\n\n", start)
    return text[start:end]


def test_refusal_rule_allows_new_requests_and_matches_runtime_prompt():
    prompt_text = PROMPT_PATH.read_text()
    instructions_text = INSTRUCTIONS_PATH.read_text()

    assert (
        "A different question later in the conversation is a NEW request" in prompt_text
    )
    assert "do not reverse your decision" not in prompt_text
    assert _refusal_section(prompt_text) == _refusal_section(instructions_text)
