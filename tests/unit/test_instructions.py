from pathlib import Path

import pytest


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


@pytest.mark.parametrize(
    ("question", "query"),
    [
        ("Does LangChain support the GPT-6 decision API?", "gpt-6 decision api"),
        ("How do I use langchain_typesafe?", "langchain_typesafe"),
    ],
)
def test_query_examples_preserve_distinguishing_terms(question, query):
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert f'"{question}" -> `query="{query}"`' in instructions
    assert "Preserve model names, API names, package identifiers" in instructions
    assert "This overrides rules 1–6 and common concept mappings" in instructions
    assert "do not shorten or normalize identifiers" in instructions


def test_denials_require_exact_term_research_and_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()
    evidence_rules = instructions.split("Evidence rules:", 1)[1].split(
        "### Writing Rules:", 1
    )[0]

    assert (
        "does not exist, is unsupported, or is not added by default" in evidence_rules
    )
    assert "at least one `search_docs_by_lang_chain` query" in evidence_rules
    assert "the user's exact distinguishing terms verbatim" in evidence_rules
    assert "This overrides query-shortening and normalization rules" in evidence_rules
    assert "Read relevant retrieved documentation" in evidence_rules
    assert (
        "with `query_docs_filesystem_docs_by_lang_chain` before answering"
        in evidence_rules
    )
    assert "establishes existence, support, or default behavior" in evidence_rules
    assert "reflect that evidence rather than deny it" in evidence_rules
    assert "never assert or deny the premise" in evidence_rules
    assert "The documentation does not state …" in evidence_rules
    assert 'query="GPT-6 decision API"' in instructions
