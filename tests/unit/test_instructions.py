from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_response_prose_matches_latest_user_message_language():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "**CRITICAL: Write all response prose in the natural language of the user's most recent message. "
        "Keep code blocks, identifiers, configuration keys, documentation titles, and URLs verbatim in their original form, "
        'and preserve the existing structure of the "Relevant docs:" footer.**'
    ) in instructions
    assert (
        "**Reply language:** All response prose matches the natural language of the user's most recent message; "
        "code blocks, identifiers, configuration keys, documentation titles, URLs, "
        'and the existing "Relevant docs:" footer structure remain unchanged'
    ) in instructions
