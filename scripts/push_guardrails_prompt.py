"""Push the local guardrails prompt to LangSmith Prompt Hub."""

import sys

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from src.prompts.guardrails_prompts import guardrails_system_prompt


def main() -> None:
    """Push the guardrails prompt in the simplest form."""
    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=guardrails_system_prompt)]
    )
    url = Client().push_prompt("public-chat-langchain-guardrails-test", object=prompt)
    sys.stdout.write(f"{url}\n")


if __name__ == "__main__":
    main()
