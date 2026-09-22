"""Push the local guardrails prompt to LangSmith Prompt Hub."""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from src.prompts.guardrails_prompts import guardrails_system_prompt


def main() -> None:
    """Push and tag the guardrails prompt for the active environment."""
    load_dotenv(".env", override=True)
    environment = (
        "staging"
        if os.getenv("LANGSMITH_HOST_PROJECT_NAME") == "immanuel-chat-langchain-test"
        or os.getenv("LANGSMITH_ENV") == "dev"
        else "production"
    )
    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=guardrails_system_prompt)]
    )
    url = Client().push_prompt(
        "langchain-ai/public-chat-langchain-guardrails-test",
        object=prompt,
        commit_tags=[environment],
    )
    sys.stdout.write(f"{url}\n")


if __name__ == "__main__":
    main()
