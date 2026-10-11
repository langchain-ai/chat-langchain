from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_named_terms_override_generic_query_rules():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()
    query_rules = instructions.split("**Query Extraction Rules (Follow EXACTLY):**")[
        1
    ].split("**Query Extraction Examples")[0]

    assert "The first `search_docs_by_lang_chain` query must retain" in query_rules
    assert (
        "import path, package/module identifier (including `_`, `-`, or `.`)"
        in query_rules
    )
    assert "model name, named API/product qualifier phrase" in query_rules
    assert "verbatim rather than replacing them with a generic concept" in query_rules
    assert "overrides core-noun-only guidance" in query_rules
    assert (
        "stripping, length, singularization, lowercase, or concept-mapping rules"
        in query_rules
    )
    assert "dcode, deepagents, langgraph, langsmith, fleet, or langserve" in query_rules
    assert "dynamic, local, self-hosted, or streaming" in query_rules


def test_named_query_examples_preserve_exact_terms():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        '"from langchain_typesafe import TypeSafeClassifier" -> `query="langchain_typesafe"`'
        in instructions
    )
    assert (
        '"does langchain support the gpt-6 decision api" -> first `query="gpt-6 decision api"`'
        in instructions
    )
    assert 'additional `query="decision models"`' in instructions
    assert "must not replace the exact-term query" in instructions


def test_named_feature_denials_require_exact_term_verification():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()
    evidence_rules = instructions.split("Evidence rules:")[1].split(
        "### Writing Rules:"
    )[0]

    assert "Before saying a named feature, package, model, or API" in evidence_rules
    assert "undocumented, unsupported, or nonexistent" in evidence_rules
    assert "recommending a different package instead of the one named" in evidence_rules
    assert (
        "at least one additional `search_docs_by_lang_chain` search" in evidence_rules
    )
    assert "retaining the user's exact relevant term" in evidence_rules
    assert (
        "even if search deduplication or round limits would otherwise stop research"
        in evidence_rules
    )
    assert (
        "If relevant hits are returned, read the most relevant documentation with `query_docs_filesystem_docs_by_lang_chain`"
        in evidence_rules
    )
    assert "report only that you could not find documentation" in evidence_rules
    assert (
        "search failure alone does not establish nonexistence or lack of support"
        in evidence_rules
    )
    assert "Technical claims still require retrieved evidence" in evidence_rules
