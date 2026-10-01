from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_docs_page_paths_follow_search_results():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "Build the read path exactly as `/` + the search result's `Page:` value + `.mdx`, used verbatim with no prefix added."
        in instructions
    )
    assert "LangSmith pages are at `/langsmith/<page>.mdx`" in instructions
    assert 'command="head -120 /langsmith/evaluation-concepts.mdx"' in instructions
