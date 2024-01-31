import argparse
import datetime
import functools
import json
import os
from typing import Literal, Optional, Union

import weaviate
from langchain import load as langchain_load
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableMap
from langchain.smith import RunEvalConfig
from langchain.vectorstores import Weaviate
from langsmith import Client, RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run

_PROVIDER_MAP = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
}

_MODEL_MAP = {
    "openai": "gpt-3.5-turbo-1106",
    "anthropic": "claude-2",
}


def search(search_queries, retriever: BaseRetriever):
    results = []
    for q in search_queries:
        results.extend(retriever.get_relevant_documents(q))
    return results


def create_search_queries_chain(
    retriever: BaseRetriever,
    model_provider: Union[Literal["openai"], Literal["anthropic"]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    include_question_and_chat_history=True,
) -> Runnable:
    model_name = model or _MODEL_MAP[model_provider]
    model = _PROVIDER_MAP[model_provider](model=model_name, temperature=temperature)
    output_parser = CommaSeparatedListOutputParser()

    _template = """Given the following conversation and a follow up question, generate a list of search queries within LangChain's internal documentation. Keep the total number of search queries to be less than 3, and try to minimize the number of search queries if possible. We want to search as few times as possible, only retrieving the information that is absolutely necessary for answering the user's questions.

1. If the user's question is a straightforward greeting or unrelated to LangChain, there's no need for any searches. In this case, output an empty list.

2. If the user's question pertains to a specific topic or feature within LangChain, identify up to two key terms or phrases that should be searched for in the documentation. If you think there are more than two relevant terms or phrases, then choose the two that you deem to be the most important/unique.

{format_instructions}

EXAMPLES:
    Chat History:

    Follow Up Input: Hi LangChain!
    Search Queries: 

    Chat History:
    What are vector stores?
    Follow Up Input: How do I use the Chroma vector store?
    Search Queries: Chroma vector store

    Chat History:
    What are agents?
    Follow Up Input: "How do I use a ReAct agent with an Anthropic model?"
    Search Queries: ReAct Agent, Anthropic Model

END EXAMPLES. BEGIN REAL USER INPUTS. ONLY RESPOND WITH A COMMA-SEPARATED LIST. REMEMBER TO GIVE NO MORE THAN TWO RESULTS.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Search Queries: """

    SEARCH_QUERIES_PROMPT = PromptTemplate.from_template(
        _template,
        partial_variables={
            "format_instructions": output_parser.get_format_instructions()
        },
    )

    chain_map = {
        "answer": {
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | SEARCH_QUERIES_PROMPT
        | model
        | output_parser,
    }

    if include_question_and_chat_history:
        chain_map["question"] = lambda x: x["question"]
        chain_map["chat_history"] = lambda x: x.get("chat_history", [])

    return RunnableMap(chain_map)


def create_chain(
    retriever: BaseRetriever,
    model_provider: Union[Literal["openai"], Literal["anthropic"]],
    chat_history: Optional[list] = None,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> Runnable:
    _inputs = create_search_queries_chain(retriever, model_provider, model, temperature)
    model_name = model or _MODEL_MAP[model_provider]
    model = _PROVIDER_MAP[model_provider](model=model_name, temperature=temperature)

    _template = """
    You are an expert programmer and problem-solver, tasked to answer any question about Langchain. Using the provided context, answer the user's question to the best of your ability using the resources provided.
    If you really don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
    Anything between the following markdown blocks is retrieved from a knowledge bank, not part of the conversation with the user. 
    <context>
        {context} 
    <context/>"""

    _context = {
        "context": lambda x: search(x["answer"], retriever),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x.get("chat_history", []),
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = _inputs | _context | prompt | model | StrOutputParser()

    return chain


def _get_retriever():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=client,
        index_name="LangChain_agent_docs",
        text_key="text",
        embedding=OpenAIEmbeddings(chunk_size=200),
        by_text=False,
        attributes=["source"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=3))


class CustomHallucinationEvaluator(RunEvaluator):
    @staticmethod
    def _get_llm_runs(run: Run) -> Run:
        runs = []
        for child in run.child_runs or []:
            if run.run_type == "llm":
                runs.append(child)
            else:
                runs.extend(CustomHallucinationEvaluator._get_llm_runs(child))

    def evaluate_run(
        self, run: Run, example: Example | None = None
    ) -> EvaluationResult:
        llm_runs = self._get_llm_runs(run)
        if not llm_runs:
            return EvaluationResult(key="hallucination", comment="No LLM runs found")
        if len(llm_runs) > 0:
            return EvaluationResult(
                key="hallucination", comment="Too many LLM runs found"
            )
        llm_run = llm_runs[0]
        messages = llm_run.inputs["messages"]
        langchain_load(json.dumps(messages))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Chat LangChain Simple Questions")
    parser.add_argument(
        "--search-queries-dataset-name", default="Chat LangChain Search Queries"
    )
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
    )
    chain = constructor()
    eval_config = RunEvalConfig(evaluators=["qa"], prediction_key="output")
    results = client.run_on_dataset(
        dataset_name=args.dataset_name,
        project_name=f"improved_chain {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        llm_or_chain_factory=constructor,
        evaluation=eval_config,
        tags=["improved_chain"],
        verbose=True,
    )
    eval_config_search_queries = RunEvalConfig(evaluators=["qa"])
    search_queries_constructor = functools.partial(
        create_search_queries_chain,
        retriever=retriever,
        model_provider=args.model_provider,
        include_question_and_chat_history=False,
    )
    search_queries_chain = search_queries_constructor()
    results = client.run_on_dataset(
        dataset_name=args.search_queries_dataset_name,
        project_name=f"improved_chain {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        llm_or_chain_factory=search_queries_chain,
        evaluation=eval_config_search_queries,
        tags=["improved_chain"],
        verbose=True,
    )
    print(results)
    proj = client.read_project(project_name=results["project_name"])
    print(proj.feedback_stats)
