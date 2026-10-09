from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_search_snippets_still_require_full_page_reads():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "Up to six hits with Title, Link, Page, and Content snippets limited to 200 characters"
        in instructions
    )
    assert (
        "ALWAYS follow up by reading the relevant docs pages with `query_docs_filesystem_docs_by_lang_chain`"
        in instructions
    )
