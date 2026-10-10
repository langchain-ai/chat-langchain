from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_reply_language_rule_and_checklist_are_preserved():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "**CRITICAL: Write all response prose in the natural language of the user's "
        "most recent message (if the user wrote in Chinese, answer in Chinese even "
        "when the retrieved docs are English). If the user earlier in the thread "
        "asked for a specific language, keep using it. Keep code blocks, identifiers, "
        "configuration keys, documentation titles, and URLs verbatim in their "
        'original form, and preserve the existing structure of the "Relevant docs:" footer.**'
    ) in instructions
    assert (
        "**Reply language:** All response prose matches the natural language of the "
        "user's most recent message (or the language the user explicitly requested "
        "earlier in the thread); code, identifiers, titles, URLs and the Relevant "
        "docs footer remain unchanged"
    ) in instructions
