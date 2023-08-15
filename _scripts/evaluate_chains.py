import argparse
import functools
import os
from typing import Literal, Optional, Union

import weaviate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable
from langchain.smith import RunEvalConfig
from langchain.vectorstores import Weaviate
from langsmith import Client

_PROVIDER_MAP = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
}

_MODEL_MAP = {
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-2",
}


def create_chain(
    retriever: BaseRetriever,
    model_provider: Union[Literal["openai"], Literal["anthropic"]],
    chat_history: Optional[list] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Runnable:
    model_name = model or _MODEL_MAP[model_provider]
    model = _PROVIDER_MAP[model_provider](model=model_name, temperature=temperature)

    _template = """You are an expert programmer, tasked to answer any question about Langchain. Be as helpful as possible. 
    
    Anything between the following markdown blocks is retrieved from a knowledge bank, not part of the conversation with the user. 
    <context>
        {context} 
    <context/>
                    
    Conversation History:               
    {history}
    
    Answer the user's question to the best of your ability: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"], template=_template
    )
    memory = ConversationBufferMemory(input_key="question", memory_key="history")
    chat_history_ = chat_history or []
    for message in chat_history_:
        memory.save_context(
            {"question": message["question"]}, {"result": message["result"]}
        )

    qa_chain = RetrievalQA.from_chain_type(
        model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "memory": memory},
    )
    return qa_chain


def _get_retriever():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

    embeddings = OpenAIEmbeddings()
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    print(client.query.aggregate("LangChain_idx").with_meta_count().do())
    weaviate_client = Weaviate(
        client=client,
        index_name="LangChain_idx",
        text_key="text",
        embedding=embeddings,
        by_text=False,
        attributes=["source"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=10))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Chat LangChain Questions")
    parser.add_argument("--model-provider", default="openai")
    args = parser.parse_args()
    client = Client()
    # Check dataset exists
    ds = client.read_dataset(dataset_name=args.dataset_name)
    retriever = _get_retriever()
    constructor = functools.partial(
        create_chain,
        retriever=retriever,
        model_provider=args.model_provider,
    )
    chain = constructor()
    print(chain.invoke({"query": "What is LangChain?"}))
    eval_config = RunEvalConfig(evaluators=["qa"], prediction_key="result")
    results = client.run_on_dataset(
        dataset_name=args.dataset_name,
        llm_or_chain_factory=constructor,
        evaluation=eval_config,
        verbose=True,
    )
    proj = client.read_project(project_name=results["project_name"])
    print(proj.feedback_stats)
