from langchain.chat_models import ChatOpenAI


def initialize_llm(model):
    return ChatOpenAI(
        model=model,
        streaming=True,
        temperature=0,
    )

