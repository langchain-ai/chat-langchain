import argparse
import json
import os
from typing import Optional

import weaviate

from backend.utils import sanitize_weaviate_url
from langchain import load as langchain_load
from langchain.agents import AgentExecutor, Tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.smith import RunEvalConfig, run_on_dataset
from langchain.vectorstores import Weaviate
from langsmith import Client, RunEvaluator
from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run

WEAVIATE_URL = sanitize_weaviate_url(os.environ["WEAVIATE_URL"])
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
WEAVIATE_DOCS_INDEX_NAME = os.environ.get(
    "WEAVIATE_INDEX_NAME", "LangChain_Combined_Docs_OpenAI_text_embedding_3_small"
)


def search(inp: str, callbacks=None) -> list:
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=OpenAIEmbeddings(chunk_size=200),
        by_text=False,
        attributes=["source"],
    )
    retriever = weaviate_client.as_retriever(
        search_kwargs=dict(k=3), callbacks=callbacks
    )

    docs = retriever.get_relevant_documents(inp, callbacks=callbacks)
    return [doc.page_content for doc in docs]


def get_tools():
    langchain_tool = Tool(
        name="Documentation",
        func=search,
        description="useful for when you need to refer to LangChain's documentation, for both API reference and codebase",
    )
    ALL_TOOLS = [langchain_tool]

    return ALL_TOOLS


def get_agent(llm, *, chat_history: Optional[list] = None):
    chat_history = chat_history or []
    # system_message = SystemMessage(
    #     content=(
    #         "You are an expert developer tasked answering questions about the LangChain Python package. "
    #         "You have access to a LangChain knowledge bank which you can query but know NOTHING about LangChain otherwise. "
    #         "You should always first query the knowledge bank for information on the concepts in the question. "
    #         "For example, given the following input question:\n"
    #         "-----START OF EXAMPLE INPUT QUESTION-----\n"
    #         "What is the transform() method for runnables? \n"
    #         "-----END OF EXAMPLE INPUT QUESTION-----\n"
    #         "Your research flow should be:\n"
    #         "1. Query your search tool for information on 'Runnables.transform() method' to get as much context as you can about it.\n"
    #         "2. Then, query your search tool for information on 'Runnables' to get as much context as you can about it.\n"
    #         "3. Answer the question with the context you have gathered."
    #         "For another example, given the following input question:\n"
    #         "-----START OF EXAMPLE INPUT QUESTION-----\n"
    #         "How can I use vLLM to run my own locally hosted model? \n"
    #         "-----END OF EXAMPLE INPUT QUESTION-----\n"
    #         "Your research flow should be:\n"
    #         "1. Query your search tool for information on 'run vLLM locally' to get as much context as you can about it. \n"
    #         "2. Answer the question as you now have enough context.\n\n"
    #         "Include CORRECT Python code snippets in your answer if relevant to the question. If you can't find the answer, DO NOT make up an answer. Just say you don't know. "
    #         "Answer the following question as best you can:"
    #     )
    # )
    system_message = SystemMessage(
        content=(
            "You are a helpful assistent tasked answering questions NatureAlpha's data and methodology. "
            "NatureAlpha is a sustainable fintech company. your aim is to help client better understand the methodology behind the data"
            "In short, this is the naturealpha mission: 'Through our innovative use of AI and geospatial data, NatureAlpha integrates nature risk into mainstream capital allocation decision. Ensuring that investments have a net-positive impact on the environment. Our mission is driven by the recognition that nature degradation poses a systemic risk to economic stability. With an estimated $44 trillion of economic value threatened by biodiversity declines and ecosystem collapse. NatureAlpha is redefining how natural world risks are measured and managed. Striving for a global economy where every dollar invested supports nature-positive outcomes.'"
            "You have access to a NatureAlpha knowledge bank which you can query but know NOTHING about NatureAlpha otherwise. "
            "You should always first query the knowledge bank for information on the concepts in the question. "
            "For example, given the following input question:\n"
            "-----START OF EXAMPLE INPUT QUESTION-----\n"
            "What data feeds into the naturerisk calculation? \n"
            "-----END OF EXAMPLE INPUT QUESTION-----\n"
            "Your research flow should be:\n"
            "1. Query your search tool for information on 'NatureRisk' to get as much context as you can about it.\n"
            "2. Then, query your search tool for information on 'Aggregation methods for core data' to get as much context as you can about it.\n"
            "3. Answer the question with the context you have gathered."
            "For another example, given the following input question:\n"
            "-----START OF EXAMPLE INPUT QUESTION-----\n"
            "Where does naturealpha get the data at the foundation of the footprint module? \n"
            "-----END OF EXAMPLE INPUT QUESTION-----\n"
            "Your research flow should be:\n"
            "1. Query your search tool for information on 'footprint data sources' to get as much context as you can about it. \n"
            "2. Answer the question as you now have enough context.\n\n"
            "Answer the following question as best you can:"
        )
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
    )

    memory = AgentTokenBufferMemory(
        memory_key="chat_history", llm=llm, max_token_limit=2000
    )

    for msg in chat_history:
        if "question" in msg:
            memory.chat_memory.add_user_message(str(msg.pop("question")))
        if "result" in msg:
            memory.chat_memory.add_ai_message(str(msg.pop("result")))

    tools = get_tools()

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=False,
        return_intermediate_steps=True,
    )

    return agent_executor


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


def return_results(client, llm):
    results = run_on_dataset(
        client=client,
        dataset_name=args.dataset_name,
        llm_or_chain_factory=lambda llm: get_agent(llm),
        evaluation=eval_config,
        verbose=True,
        concurrency_level=0,  # Add this to not go async
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Chat LangChain Complex Questions")
    parser.add_argument("--model-provider", default="openai")
    parser.add_argument("--prompt-type", default="chat")
    args = parser.parse_args()
    client = Client()
    # Check dataset exists
    ds = client.read_dataset(dataset_name=args.dataset_name)

    llm = ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True, temperature=0)

    eval_config = RunEvalConfig(evaluators=["qa"], prediction_key="output")
    results = run_on_dataset(
        client,
        dataset_name=args.dataset_name,
        llm_or_chain_factory=lambda x: get_agent(llm),
        evaluation=eval_config,
        verbose=False,
        concurrency_level=0,  # Add this to not go async
        tags=["agent"],
        input_mapper=lambda x: x["question"],
    )
    print(results)

    proj = client.read_project(project_name=results["project_name"])
    print(proj.feedback_stats)
