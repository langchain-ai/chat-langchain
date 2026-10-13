"""Fail CI when the production Hub guardrails differ from the repository."""

import sys

from dotenv import load_dotenv
from langsmith import Client

from src.utils.guardrails_prompt import verify_guardrails_prompt


def main() -> None:
    """Verify the production system prompt hash against the repository."""
    load_dotenv(".env", override=True)
    template = Client().pull_prompt(
        "langchain-ai/public-chat-langchain-guardrails-test:production"
    )
    prompt_hash = verify_guardrails_prompt(
        template.format_messages(messages=[])[0].content
    )
    sys.stdout.write(
        f"Guardrails production prompt matches repository: {prompt_hash}\n"
    )


if __name__ == "__main__":
    main()
