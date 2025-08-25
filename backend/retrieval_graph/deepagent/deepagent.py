import os
from typing import Literal
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from typing import Optional
from deepagents import create_deep_agent, SubAgent, DeepAgentState
from backend.retrieval_graph.deepagent.prompts import RAG_TOOL_DESCRIPTION, DEEP_AGENT_DEFAULT_INSTRUCTIONS
from langchain_core.tools import tool
from backend.retrieval import make_retriever
from langchain_core.documents import Document

# TODO: Make this configurable eventually?
config = {
    "configurable": {
        "embedding_model": "openai/text-embedding-3-small",
        "retriever_provider": "weaviate",
    }
}

@tool(description=RAG_TOOL_DESCRIPTION)
async def guide_rag_search(query: str) -> str:
    with make_retriever(config) as retriever:
        response = await retriever.ainvoke(query, config)
    return {"documents": response}

class AgentConfig(BaseModel):
    instructions: Optional[str] = DEEP_AGENT_DEFAULT_INSTRUCTIONS
    subagents: Optional[list[dict]] = []

class StateSchema(DeepAgentState):
    documents: list[Document]

def deep_agent_factory(config: RunnableConfig):
    cfg = AgentConfig(**config.get("configurable", {}))
    return create_deep_agent(
        [guide_rag_search],
        cfg.instructions,
        subagents=cfg.subagents,
        config_schema=AgentConfig
    ).with_config({"recursion_limit": 100})