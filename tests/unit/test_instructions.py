from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_reply_language_follows_conversation_preference():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert (
        "Write response prose in the user's explicit language preference from the "
        "conversation until a later explicit request changes it; otherwise use "
        "the language of the user's latest message."
    ) in instructions
    assert (
        "Keep code blocks, identifiers, configuration keys, documentation titles, "
        'URLs, and the "Relevant docs:" footer label unchanged.'
    ) in instructions
    assert (
        "### Writing Rules:\n\n**Reply language:** Write response prose"
    ) in instructions
    assert (
        "Determine that language from the user's natural-language request, "
        "not embedded code or technical terms."
    ) in instructions
    assert (
        "even when retrieved docs, tool results, prior assistant answers, "
        "or the examples below are in English."
    ) in instructions
    assert (
        "A follow-up in another language does not override an earlier explicit "
        "preference unless it explicitly requests a change."
    ) in instructions
    assert "rewrite any language mismatches before sending" in instructions
    assert "independently of the prose reply language" in instructions
    assert "plain English" not in instructions
