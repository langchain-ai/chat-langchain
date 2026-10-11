from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert (
        'open with a bold sentence saying "The documentation does not state whether ..."'
        in instructions
    )
    assert "require a quoted source from the retrieved content" in instructions


def test_negative_capability_claims_require_evidence_in_how_to_answers():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "Negative capability claims also require this support, including in how-to answers"
        in instructions
    )
    for claim in (
        "X does not automatically add Y",
        "X does not provide Y",
        "X does not support Y",
        "there is nothing to turn off / no removal is necessary",
    ):
        assert claim in instructions
    assert "never contradict retrieved content" in instructions
    assert (
        "Scope conclusions to the documented API, language, and provider"
        in instructions
    )
    assert (
        "JavaScript default prompt caching does not establish Python Bedrock middleware behavior"
        in instructions
    )


def test_verdicts_require_applicable_current_turn_reads():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "A yes/no verdict or negative capability claim requires supporting content read "
        "in the current turn through `query_docs_filesystem_docs_by_lang_chain` "
        "or `get_support_article_content`"
    ) in instructions
    assert "A `search_docs_by_lang_chain` snippet alone is insufficient" in instructions
    assert "an unrelated read does not establish the verdict" in instructions
    assert (
        "cite the closest documented facts instead of inferring an answer"
        in instructions
    )
    assert (
        "Documentation or support content from an earlier turn is not evidence for a new question"
        in instructions
    )


def test_bold_opening_rules_are_conditional_on_current_turn_reads():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "answering the core question directly when current-turn reads support it"
        in instructions
    )
    assert (
        "First sentence is bold and answers the question when current-turn reads support it"
        in instructions
    )
    assert "answer only when current-turn reads support it" in instructions
    assert "first sentence answers the question" not in instructions
