from pathlib import Path

from src.prompts.docs_agent_prompt import docs_agent_prompt

RESPONSE_LANGUAGE_RULE = (
    "Write all prose in the same natural language as the user's most recent message. "
    "If the user switches language mid-thread, switch with them. "
    "Keep code blocks, API names, identifiers, CLI commands, file paths, and cited "
    "documentation URLs unchanged in their original form. "
    "When quoting documentation verbatim, preserve the quote and add a short "
    "translated gloss."
)


def test_local_prompts_include_response_language_rule():
    instructions = Path("instructions.md").read_text()

    assert instructions.count(RESPONSE_LANGUAGE_RULE) == 2
    assert docs_agent_prompt.count(RESPONSE_LANGUAGE_RULE) == 2
