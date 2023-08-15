import argparse
import functools
import os
from typing import Literal, Optional, Union

import weaviate
from langchain import prompts
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
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


def _get_prompt(prompt_type: str) -> prompts.BasePromptTemplate:
    if prompt_type == "completion":
        _template = """You are an expert programmer, tasked to answer any question about Langchain. Be as helpful as possible. 
        
Anything between the following markdown blocks is retrieved from a knowledge bank, not part of the conversation with the user. 
<context>
    {context} 
<context/>
                
Conversation History:               
{history}

Answer the user's question to the best of your ability: {question}
Helpful Answer:"""

        return prompts.PromptTemplate(
            input_variables=["history", "context", "question"], template=_template
        )
    else:
        return prompts.ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert programmer, tasked with answering any question about Langchain. Be as helpful as possible.",
                ),
                prompts.MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
                (
                    "system",
                    "Respond to the user's question as best and truthfully as you are able."
                    " You can choose to use the following retrieved information if it is relevant.",
                ),
                ("system", "<retrieved_context>\n{context}\n</retrieved_context>"),
            ]
        )


def create_chain(
    retriever: BaseRetriever,
    model_provider: Union[Literal["openai"], Literal["anthropic"]],
    chat_history: Optional[list] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
    prompt_type: Union[Literal["chat"], Literal["completion"]] = "chat",
) -> Runnable:
    model_name = model or _MODEL_MAP[model_provider]
    model = _PROVIDER_MAP[model_provider](model=model_name, temperature=temperature)

    prompt = _get_prompt(prompt_type)
    return_messages = True if prompt_type == "chat" else False

    memory = ConversationBufferMemory(
        input_key="question", memory_key="history", return_messages=return_messages
    )
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
    parser.add_argument("--prompt-type", default="chat")
    args = parser.parse_args()
    client = Client()
    # Check dataset exists
    ds = client.read_dataset(dataset_name=args.dataset_name)
    retriever = _get_retriever()
    constructor = functools.partial(
        create_chain,
        retriever=retriever,
        model_provider=args.model_provider,
        prompt_type=args.prompt_type,
    )
    chain = constructor()
    eval_config = RunEvalConfig(evaluators=["qa"], prediction_key="result")
    results = client.run_on_dataset(
        dataset_name=args.dataset_name,
        llm_or_chain_factory=constructor,
        evaluation=eval_config,
        verbose=True,
    )
    proj = client.read_project(project_name=results["project_name"])
    print(proj.feedback_stats)
