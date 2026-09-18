from __future__ import annotations

import re
from pathlib import Path

from src.prompts.docs_agent_prompt import docs_agent_prompt

BOUNDARIES = """## Boundaries

**IMPORTANT: You cannot see the user's repository, private code, logs, or installed build. When a user asks whether behaviour in THEIR code is intended, or asks you to adjudicate their internal symbols, say plainly that you cannot inspect their code, and answer only about the public library contract you retrieved on this turn. Never assert what the author of code you have not read intended.**

**IMPORTANT: You are a documentation assistant, not a LangChain maintainer. Never use first-person plural for the LangChain team, never commit to future releases, roadmap items, deprecations, or documentation changes, and never say you will amend anything. If asked for a roadmap or product commitment, state that you can only report what current documentation says and point the user to support.**"""


def test_docs_agent_boundaries_are_present_and_mirrored():
    instructions = Path("instructions.md").read_text()

    assert BOUNDARIES in docs_agent_prompt
    assert BOUNDARIES in instructions
    prompt_boundaries = re.search(
        r"## Boundaries\n\n.*?(?=\n\nDo not assume)",
        docs_agent_prompt,
        re.DOTALL,
    )
    instructions_boundaries = re.search(
        r"## Boundaries\n\n.*?(?=\n\nDo not assume)",
        instructions,
        re.DOTALL,
    )

    assert prompt_boundaries is not None
    assert instructions_boundaries is not None
    assert prompt_boundaries.group() == instructions_boundaries.group()
