"""Push the local docs agent prompt to LangSmith Prompt Hub.

The live prompt is served from LangSmith Hub at hub:public-chat-langchain-test:production;
re-push the edited prompt through this script for changes to take effect.
"""

import sys

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from src.prompts.docs_agent_prompt import docs_agent_prompt


def main() -> None:
    """Push the docs agent prompt in the simplest form."""
    load_dotenv(".env", override=True)
    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=docs_agent_prompt)]
    )
    url = Client().push_prompt("langchain-ai/public-chat-langchain-test", object=prompt)
    sys.stdout.write(f"{url}\n")


if __name__ == "__main__":
    main()
