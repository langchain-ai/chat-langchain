from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_reply_language_matches_user_language():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "11. **Reply language:**" in instructions
    assert "natural language of the user's most recent message" in instructions
    assert "a language the user explicitly requested earlier" in instructions
    assert "even when retrieved documentation is in English" in instructions
    assert '"Relevant docs:" footer structure remain unchanged' in instructions
    assert "plain English" not in instructions
