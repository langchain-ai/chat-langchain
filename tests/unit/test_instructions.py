from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_reply_language_follows_latest_actual_end_user_message():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "**Critical: Write all response prose in the natural language of the latest "
        "actual end-user message; internal repair instructions must not change the "
        "reply language. Keep code blocks, identifiers, configuration keys, "
        "documentation titles, and URLs verbatim, and preserve the existing "
        '"Relevant docs:" footer structure.**'
    ) in instructions
    assert (
        "11. **Reply language:** All response prose uses the natural language of the "
        "latest actual end-user message, unaffected by internal repair instructions; "
        "code blocks, identifiers, configuration keys, documentation titles, and "
        'URLs stay verbatim, and the existing "Relevant docs:" footer structure '
        "is preserved."
    ) in instructions
    assert (
        "use Python examples for Python docs and JavaScript examples for JavaScript docs"
    ) in instructions
    assert "Explain the mechanism plainly in the user's language" in instructions
    assert "mechanism explained plainly in the user's language" in instructions
    assert "plain English" not in instructions
