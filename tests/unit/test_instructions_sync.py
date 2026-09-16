from pathlib import Path

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_instructions_matches_source_prompt():
    instructions_path = Path(__file__).parents[2] / "instructions.md"

    assert instructions_path.read_text() == docs_agent_prompt
