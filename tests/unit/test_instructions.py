from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_reply_language_matches_user_while_preserving_technical_content():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "Write all response prose in the natural language of the user's most recent message"
        in instructions
    )
    assert (
        "or the language the user explicitly asked for earlier in the conversation"
        in instructions
    )
    assert (
        "Keep code blocks, identifiers, configuration keys, documentation titles, and URLs verbatim"
        in instructions
    )
    assert (
        'preserve the existing structure of the "Relevant docs:" footer' in instructions
    )
    assert "plain English" not in instructions

    checklist = instructions.split("Before sending your response, verify:")[1]
    assert (
        "**Reply language:** All response prose matches the natural language of the user's most recent message"
        in checklist
    )
    assert "or the language they explicitly requested earlier" in checklist
    assert (
        'code blocks, identifiers, configuration keys, documentation titles, URLs, and the existing "Relevant docs:" footer structure remain unchanged'
        in checklist
    )


def test_support_search_requires_query_from_user_question():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    support_search = instructions.split("### 4. `search_support_articles`")[1].split(
        "### 5."
    )[0]
    assert "Always pass a `query` built from the user's question" in support_search
    assert (
        "ranked article IDs, titles, URLs, collections, and snippets" in support_search
    )

    research = instructions.split("### Step 1: Research Documentation and Support KB")[
        1
    ].split("### Step 2:")[0]
    assert (
        "Call `search_support_articles` once with a `query` built from the user's question"
        in research
    )
