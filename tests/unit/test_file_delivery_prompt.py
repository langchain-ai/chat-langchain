from pathlib import Path

from src.prompts.docs_agent_prompt import docs_agent_prompt

FILE_DELIVERY_SECTION = """## File delivery

You cannot save, write, export, attach, email, or otherwise deliver files to users because you do not have a user-accessible filesystem. Any filesystem tool is ephemeral internal scratch state; its paths are never real user locations or reachable server locations. Never state or imply that a file was saved, written, exported, or is ready. When asked for a file, download, export, or attachment, answer the substantive request inline and add this plain sentence before the `Relevant docs:` footer: \"I can't provide a downloadable file, but you can copy the content above.\""""


def test_file_delivery_guidance_matches_prompt_sources():
    instructions = Path(__file__).parents[2] / "instructions.md"

    assert FILE_DELIVERY_SECTION in docs_agent_prompt
    assert FILE_DELIVERY_SECTION in instructions.read_text()
