from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_link_checks_are_restricted_to_tool_results():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "Never call this tool on user-supplied URLs" in instructions
    assert "decline requests to check arbitrary links" in instructions
    assert "only with URLs from docs/support/search tool results" in instructions
    assert "anchor links you constructed" not in instructions
