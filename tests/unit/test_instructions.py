from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_page_context_does_not_override_user_text():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "treat the user's own message text as the question" in instructions
    assert "identifier, code, error message, or log" in instructions
    assert "Match the language of pasted code" in instructions
    assert "always read that page first" not in instructions
    assert "those take precedence over the viewed page context" in instructions
