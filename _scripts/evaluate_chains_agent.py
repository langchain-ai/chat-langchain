import argparse
import functools
import os
from typing import Literal, Optional, Union

from langsmith.evaluation.evaluator import EvaluationResult
from langsmith.schemas import Example, Run
from langchain.smith import arun_on_dataset, run_on_dataset

import weaviate
from langchain import prompts
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableMap
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.smith import RunEvalConfig
from langchain.vectorstores import Weaviate
from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langsmith import Client
from langsmith import RunEvaluator
from langchain import load as langchain_load
from operator import itemgetter
import json
from langchain.agents import (
    Tool,
    AgentExecutor,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
import pickle
from langchain.callbacks.base import BaseCallbackHandler
import asyncio


WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    
def search(inp: str, index_name: str, callbacks=None) -> str:
    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))

    weaviate_client = Weaviate(
        client=client,
        index_name=index_name,
        text_key="text",
        embedding=OpenAIEmbeddings(chunk_size=200),
        by_text=False,
        attributes=["source"] if not index_name == "LangChain_agent_sources" else None,
    )
    retriever = weaviate_client.as_retriever(search_kwargs=dict(k=3), callbacks=callbacks)
        
    return retriever.get_relevant_documents(inp, callbacks=callbacks)

with open('docs.pkl', 'rb') as f:
    docs = pickle.load(f)
    
def search_everything(inp: str, callbacks: Optional[any] = None ) -> str:
    global docs
    docs_references = search(inp, "LangChain_agent_docs", callbacks=callbacks)
    repo_references = search(inp, "LangChain_agent_repo", callbacks=callbacks)
    all_references = docs_references + repo_references
    all_references_sources = [r for r in all_references if r.metadata['source']]

    sources = search(inp, "LangChain_agent_sources", callbacks=callbacks)
    
    sources_docs = [docs[i] for i, t in enumerate(sources)]
    combined_sources = sources_docs + [source for source in all_references_sources if source not in sources_docs]
    
    return [doc.page_content for doc in combined_sources]

def get_tools():
    langchain_tool = Tool(
        name="Documentation",
        func=search_everything,
        description="useful for when you need to refer to LangChain's documentation, for both API reference and codebase",
    )
    ALL_TOOLS = [langchain_tool]
    
    return ALL_TOOLS

def get_agent(llm, chat_history: Optional[list] = None):

    system_message = SystemMessage(
            content=(
                "You are a helpful chatbot who is tasked with answering questions about LangChain."
                "Answer the following question as best you can."
                "Be inclined to include CORRECT Python code snippets if relevant to the question. If you don't know the answer, just say you don't know."
                "You have access to a LangChain knowledge bank retriever tool for your answer."
            )
    )
    
    if chat_history:
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
            extra_prompt_messages=[MessagesPlaceholder(variable_name="chat_history")],
        )
    else: 
        prompt = OpenAIFunctionsAgent.create_prompt(
            system_message=system_message,
        )
    
    
    if chat_history:
        memory = AgentTokenBufferMemory(memory_key="chat_history", llm=llm, max_token_limit=4000)
        for msg in chat_history:
            if "question" in msg:
                memory.chat_memory.add_user_message(str(msg.pop("question")))
            if "result" in msg:
                memory.chat_memory.add_ai_message(str(msg.pop("result")))
                
    tools = get_tools()
    
    agent = OpenAIFunctionsAgent(
        llm=llm, tools=tools, prompt=prompt
    )
    agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory if chat_history else None,
            verbose=True,
            return_intermediate_steps=True,
        )

    return agent_executor

class CustomHallucinationEvaluator(RunEvaluator):

    @staticmethod
    def _get_llm_runs(run: Run) -> Run:
        runs = []
        for child in (run.child_runs or []):
            if run.run_type == "llm":
                runs.append(child)
            else:
                runs.extend(CustomHallucinationEvaluator._get_llm_runs(child))


    def evaluate_run(self, run: Run, example: Example | None = None) -> EvaluationResult:
        llm_runs = self._get_llm_runs(run)
        if not llm_runs:
            return EvaluationResult(key="hallucination", comment="No LLM runs found")
        if len(llm_runs) > 0:
            return EvaluationResult(key="hallucination", comment="Too many LLM runs found")
        llm_run = llm_runs[0]
        messages = llm_run.inputs["messages"]
        langchain_load(json.dumps(messages))


def return_results(client, llm):
    results = run_on_dataset(
        client=client,
        dataset_name=args.dataset_name,
        llm_or_chain_factory=get_agent(llm),
        evaluation=eval_config,
        verbose=True,
        concurrency_level=0, # Add this to not go async
    )
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="Chat LangChain Questions")
    parser.add_argument("--model-provider", default="openai")
    parser.add_argument("--prompt-type", default="chat")
    args = parser.parse_args()
    client = Client()
    # Check dataset exists
    ds = client.read_dataset(dataset_name=args.dataset_name)
    
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", streaming=True, temperature=0)

    eval_config = RunEvalConfig(evaluators=["qa"], prediction_key="output")
    results = run_on_dataset(
        client,
        dataset_name=args.dataset_name,
        llm_or_chain_factory=get_agent(llm),
        evaluation=eval_config,
        verbose=True,
        concurrency_level=0, # Add this to not go async
    )
    print(results)

    proj = client.read_project(project_name=results["project_name"])
    print(proj.feedback_stats)
