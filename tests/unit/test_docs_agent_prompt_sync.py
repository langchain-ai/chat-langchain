import re
import unicodedata
from pathlib import Path

from src.prompts.docs_agent_prompt import docs_agent_prompt

CAPABILITY_MARKER = "**You CANNOT open, create, file, or submit support tickets"


def _extract_capability_paragraph(text: str) -> str:
    paragraph = next(
        paragraph
        for paragraph in text.split("\n\n")
        if paragraph.startswith(CAPABILITY_MARKER)
    )
    normalized = unicodedata.normalize("NFKC", paragraph)
    normalized = normalized.translate(
        str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "—": "-", "–": "-"})
    )
    return re.sub(r"\s+", " ", normalized).strip()


def test_capability_paragraph_matches_instructions() -> None:
    instructions = Path(__file__).parents[2] / "instructions.md"

    assert _extract_capability_paragraph(
        instructions.read_text()
    ) == _extract_capability_paragraph(docs_agent_prompt)
