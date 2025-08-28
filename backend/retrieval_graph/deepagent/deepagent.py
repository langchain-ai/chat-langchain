import os
import json
import sqlite3
from typing import Literal
from pydantic import BaseModel
from langchain_core.runnables import RunnableConfig
from typing import Optional
from deepagents import create_deep_agent, SubAgent, DeepAgentState
from backend.retrieval_graph.deepagent.prompts import RAG_TOOL_DESCRIPTION, DEEP_AGENT_DEFAULT_INSTRUCTIONS, LANGSMITH_API_TOOL_DESCRIPTION
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
    return {"documents": [{"page_content": doc.page_content, "source": doc.metadata["source"], "title": doc.metadata["title"]} for doc in response]}


@tool(description=LANGSMITH_API_TOOL_DESCRIPTION)
async def langsmith_api_search(match_string: str) -> str:
    conn = sqlite3.connect("api_sdk_docs.db")
    cur = conn.cursor()
    
    cur.execute("""
        SELECT title, url, content 
        FROM docs 
        WHERE domain = 'langsmith_api' 
        AND title LIKE ?
        LIMIT 10
    """, (f"%{match_string}%",))
    
    results = cur.fetchall()
    conn.close()
    
    if not results:
        return {"message": f"No results found for '{match_string}'"}
    
    return {
        "results": [{"title": title, "url": url, "content": content} 
                   for title, url, content in results],
        "count": len(results)
    }


class AgentConfig(BaseModel):
    instructions: Optional[str] = DEEP_AGENT_DEFAULT_INSTRUCTIONS
    subagents: Optional[list[dict]] = []

class StateSchema(DeepAgentState):
    documents: list[Document]

def deep_agent_factory(config: RunnableConfig):
    cfg = AgentConfig(**config.get("configurable", {}))
    return create_deep_agent(
        [guide_rag_search, langsmith_api_search],
        cfg.instructions,
        subagents=cfg.subagents,
        config_schema=AgentConfig
    ).with_config({"recursion_limit": 100})

deep_agent = deep_agent_factory(config)