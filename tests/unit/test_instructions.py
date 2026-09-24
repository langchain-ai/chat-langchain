from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_deep_agents_products_are_disambiguated():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "Deep Agents names three distinct products" in instructions
    assert "definitional or overview question about \"deep agents\" or \"deepagents\"" in instructions
    assert "Use `/oss/deepagents/code/` sources only when the user names the CLI" in instructions
    assert "questions mentioning hosting, \"managed\", or LangSmith" in instructions
