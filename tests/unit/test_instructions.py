from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_support_articles_do_not_override_current_official_documentation():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "follow the official documentation" in instructions
    assert "legacy aliases" in instructions
    assert "LANGSMITH_API_KEY" in instructions
    assert (
        "never cite a documentation page for a claim that page does not contain"
        in instructions
    )
