from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_answers_target_the_latest_user_message():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "Always answer the most recent user message" in instructions
    assert "Refer to them only if the latest message refers to them" in instructions
