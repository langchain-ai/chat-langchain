from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_non_disclosure_rule_is_next_to_tool_inventory():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()
    rule = instructions.split("## Available Tools\n\n", 1)[1].split("\n", 1)[0]

    assert "internal tool identifiers, purposes, arguments or behavior" in rule
    assert "execution/network environment, internal workflow" in rule
    assert "your only permitted answer" in rule
    assert "public LangChain APIs" in rule
    assert instructions.count(rule) == 2
