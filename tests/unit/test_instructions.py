from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_link_checks_are_limited_to_retrieved_citation_urls():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "only validates documentation/support URLs you plan to cite from search results"
        in instructions
    )
    assert (
        "never use it to test the reachability of URLs supplied by the user"
        in instructions
    )
