from pathlib import Path

import pytest
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate


@pytest.fixture(params=["local", "rendered"])
def instructions(request):
    content = (Path(__file__).parents[2] / "instructions.md").read_text()
    if request.param == "rendered":
        prompt = ChatPromptTemplate.from_messages([SystemMessage(content=content)])
        return prompt.format_messages()[0].content
    return content


def test_yes_no_answers_require_retrieved_evidence(instructions):
    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "supports the factual premise and verdict" in instructions
    assert "otherwise, never assert or deny the premise" in instructions
    assert "the documentation does not state the claim" in instructions
    assert (
        "Documentation or support content from an earlier turn is not evidence"
        in instructions
    )
    assert "require a quoted source from the retrieved content" in instructions
    assert (
        "never reuse a real documentation URL as support for a claim its content does not make"
        in instructions
    )


def test_denials_require_current_turn_evidence_and_exact_search(instructions):
    assert (
        "Never state that a package, function, parameter, or default behavior does not exist or is not used"
        in instructions
    )
    assert (
        "unless a retrieved page or support article from this turn explicitly supports that denial; quote the supporting text"
        in instructions
    )
    assert (
        "If the user names a package/module/API, search that exact token with search_docs_by_lang_chain before any denial"
        in instructions
    )
    assert "earlier-turn content cannot support a new verdict or denial" in instructions


def test_query_extraction_preserves_named_identifiers(instructions):
    assert (
        "Keep user-named identifiers verbatim, including casing and underscores"
        in instructions
    )
    assert (
        "This overrides rules 1-6, including the generic 1-2-word limit" in instructions
    )
    assert (
        '"from langchain_typesafe import TypeSafeClassifier" -> `query="langchain_typesafe"`'
        in instructions
    )
    assert (
        '"What is middleware in LangChain?" -> `query="LangChain middleware"`'
        in instructions
    )
    assert (
        '"Deployment guide for LangGraph" -> `query="LangGraph deployment"`'
        in instructions
    )
    assert (
        '"LangSmith tracing in Python?" -> `query="LangSmith python tracing"`'
        in instructions
    )
    assert (
        '"Stream from subagents?" -> Search "streaming" + "subgraphs"'
        not in instructions
    )


def test_bold_opening_requires_evidence_in_best_practices(instructions):
    best_practices = instructions.split("## Best Practices", 1)[1]
    assert (
        "Start with a bold answer only when retrieved evidence supports it"
        in best_practices
    )
    assert (
        "otherwise, state that the documentation does not state the claim"
        in best_practices
    )
    assert "**Start with bold answer**" not in best_practices
