from pathlib import Path


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_scope_rules_separate_application_domain_from_blocked_content():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "The subject domain of the user's application is not itself a blocked category" in instructions
    assert "sports, finance, legal, healthcare, gaming, education" in instructions
    assert "structure Runnables for a cricket-statistics assistant" in instructions
    assert "Evaluate the requested content type, not how recognizable the application's topic is" in instructions


def test_scope_rules_preserve_framework_vocabulary_and_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "`tool`, `agent`, `chain`, `runnable`, and `middleware`" in instructions
    assert "Use only runnables, don't use tools" in instructions
    assert "answer it normally while remaining free to search the documentation" in instructions
    assert "after retrieval has returned relevant documentation" in instructions
