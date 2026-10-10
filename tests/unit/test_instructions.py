from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_reply_language_rules_preserve_explicit_requests():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    rules = [
        next(
            line
            for line in instructions.splitlines()
            if line.startswith("**CRITICAL: Reply language:")
        ),
        next(
            line
            for line in instructions.splitlines()
            if line.startswith("11. **Reply language:**")
        ),
    ]

    for rule in rules:
        assert "most recent explicit response-language request" in rule
        assert "until they change or withdraw it" in rule
        assert (
            "otherwise, use the natural language of their most recent message" in rule
        )
        assert (
            "Explicit response-language requests take precedence over the language of subsequent questions"
            in rule
        )
        assert "response prose" in rule
        assert (
            "code blocks, identifiers, configuration keys, documentation titles, and URLs verbatim in their original form"
            in rule
        )
        assert 'preserve the existing structure of the "Relevant docs:" footer' in rule

    assert "even when retrieved documentation is in English" in rules[0]
    assert "regardless of the documentation's language" in rules[1]
