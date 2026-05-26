"""Push the local docs agent prompt to LangSmith Prompt Hub."""

import sys

from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from src.prompts.docs_agent_prompt import docs_agent_prompt


def main() -> None:
    """Push the docs agent prompt in the simplest form."""
    prompt = ChatPromptTemplate.from_messages([SystemMessage(content=docs_agent_prompt)])
    url = Client().push_prompt("public-chat-langchain-test", object=prompt)
    sys.stdout.write(f"{url}\n")


if __name__ == "__main__":
    main()
