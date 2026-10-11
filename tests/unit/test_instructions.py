from pathlib import Path

import pytest

from scripts import push_docs_agent_prompt


@pytest.fixture(params=["local", "rendered"])
def instructions(request, monkeypatch):
    content = (Path(__file__).parents[2] / "instructions.md").read_text()
    if request.param == "rendered":
        published_prompts = []

        class FakeClient:
            def push_prompt(self, identifier, *, object):
                assert identifier == "langchain-ai/public-chat-langchain-test"
                published_prompts.append(object)
                return "prompt-url"

        monkeypatch.setattr(push_docs_agent_prompt, "Client", FakeClient)
        monkeypatch.setattr(
            push_docs_agent_prompt, "load_dotenv", lambda *args, **kwargs: None
        )
        push_docs_agent_prompt.main()
        assert len(published_prompts) == 1
        rendered_content = published_prompts[0].format_messages()[0].content
        assert rendered_content == content
        return rendered_content
    return content


def test_yes_no_answers_require_retrieved_evidence(instructions):
    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "supports the factual premise and verdict" in instructions
    assert "otherwise, never assert or deny the premise" in instructions
    assert "I couldn't confirm X in the documentation I read" in instructions
    assert (
        "apply the same check before correcting the premise of a how-to question"
        in instructions
    )
    assert (
        "Lead with a direct, bolded verdict only when it does and quote the supporting passage"
        in instructions
    )
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


def test_documented_behavior_takes_precedence_over_unsupported_corrections(
    instructions,
):
    assert (
        "Explicitly documented behavior takes precedence over assumptions about implementation or terminology"
        in instructions
    )
    assert (
        "a different feature name or mechanism is not evidence that the requested behavior is absent"
        in instructions
    )
    assert (
        "A documented alternative is not evidence against the user's named API"
        in instructions
    )
    assert (
        "Missing search hits, omitted details, and partial page reads do not prove nonexistence"
        in instructions
    )
    assert (
        "Before drafting, match each premise, verdict, and proposed correction to an explicit passage"
        in instructions
    )


def test_query_extraction_preserves_named_identifiers(instructions):
    assert (
        "Keep user-named identifiers verbatim, including casing and underscores"
        in instructions
    )
    assert (
        "This overrides rules 1-6, including the generic 1-2-word limit" in instructions
    )
    assert (
        "Named identifiers and qualifiers follow rule 7, not this limit" in instructions
    )
    assert "Search user-named packages/modules/APIs verbatim" in instructions
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
        "otherwise, say what you could not confirm in the documentation you read"
        in best_practices
    )
    assert "**Start with bold answer**" not in best_practices


def test_format_validation_cannot_override_evidence(instructions):
    assert "bold formatting never requires an unsupported verdict" in instructions
    assert "Check the evidence rules before bolding the first sentence" in instructions
    assert "Check: Bold opening sentence (starts with **)" not in instructions
    assert "**Bold opening:** First sentence starts with" not in instructions
    assert (
        "otherwise state that the documentation does not state the claim"
        not in instructions
    )
