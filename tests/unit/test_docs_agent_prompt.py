from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.prompts.docs_agent_prompt import docs_agent_prompt


def test_composed_docs_agent_prompt_has_support_boundaries():
    prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content=docs_agent_prompt)]
    ).format_messages()[0].content

    assert "CANNOT open, create, file, or submit support tickets" in prompt
    assert "you ARE the support system" not in prompt
