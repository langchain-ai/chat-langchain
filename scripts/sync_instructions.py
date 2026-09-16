"""Synchronize the generated prompt mirror."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    """Write the maintained prompt to its generated mirror."""
    from src.prompts.docs_agent_prompt import docs_agent_prompt

    Path(__file__).resolve().parents[1].joinpath("instructions.md").write_text(
        docs_agent_prompt
    )


if __name__ == "__main__":
    main()
