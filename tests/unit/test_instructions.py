from pathlib import Path


def test_bold_opening_requires_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "every factual bottom line" in instructions
    assert "behavioral results, return values, import paths, and defaults" in instructions
    assert "Retrieved documentation outranks model memory" in instructions
    assert "include a short quote from that passage" in instructions
    assert "the documentation does not state the claim" in instructions
