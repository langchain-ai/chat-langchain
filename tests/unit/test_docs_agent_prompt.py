from pathlib import Path

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_docs_agent_prompt_matches_deployed_instructions():
    instructions_path = Path(__file__).parents[2] / "instructions.md"

    assert docs_agent_prompt == instructions_path.read_text()
