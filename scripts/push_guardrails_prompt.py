"""Push the local guardrails prompt to LangSmith Prompt Hub."""

import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from src.prompts.guardrails_prompts import guardrails_system_prompt


def main() -> None:
    """Push the guardrails prompt and update its deployment tag."""
    load_dotenv(".env", override=True)
    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=guardrails_system_prompt)]
    )
    client = Client()
    targeting_staging = (
        os.getenv("LANGSMITH_HOST_PROJECT_NAME") == "immanuel-chat-langchain-test"
        or os.getenv("LANGSMITH_ENV") == "dev"
    )
    commit_tags = ["production"]
    if targeting_staging:
        commit_tags.append("staging")
    prompt_identifier = "langchain-ai/public-chat-langchain-guardrails-test"
    url = client.push_prompt(
        prompt_identifier,
        object=prompt,
        commit_tags=commit_tags,
    )
    commit_hash = client._get_latest_commit_hash(prompt_identifier)
    sys.stdout.write(f"{commit_hash}\n{url}\n")


if __name__ == "__main__":
    main()
