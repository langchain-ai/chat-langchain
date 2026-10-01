from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_docs_filesystem_paths_match_search_page_values():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "The filesystem path is `/` + the exact `Page:` value" in instructions
    assert "`Page: langsmith/evaluators` becomes `/langsmith/evaluators.mdx`" in instructions
