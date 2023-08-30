"""Main entrypoint for the app."""
import os
from typing import Optional

import weaviate
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.vectorstores import Weaviate
from langsmith import Client
from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator
from langchain.agents import (
    Tool,
    AgentExecutor,
)
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
import pickle
from langchain.callbacks.base import BaseCallbackHandler

client = Client()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(callbacks=[run_collector])
run_id = None
feedback_recorded = False

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
                "You are a helpful chatbot who is tasked with answering questions about LangChain. "
                "Answer the following question as best you can. "
                "Be inclined to include CORRECT Python code snippets if relevant to the question. If you can't find the answer, just say you don't know. "
                "You have access to a LangChain knowledge bank retriever tool for your answer. "
                "You know NOTHING about LangChain's codebase."
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
    
    memory = AgentTokenBufferMemory(memory_key="chat_history", llm=llm, max_token_limit=2000)
    
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
            memory=memory,
            verbose=True,
            return_intermediate_steps=True,
        )

    return agent_executor


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""
    # https://gist.github.com/mortymike/70711b028311681e5f3c6511031d5d43

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()
    
@app.post("/chat")
async def chat_endpoint(request: Request):
    global run_id, feedback_recorded, trace_url
    run_id = None
    trace_url = None
    feedback_recorded = False
    run_collector.traced_runs = []

    data = await request.json()
    question = data.get("message")
    chat_history = data.get("history", [])
    conversation_id = data.get("conversation_id")
    
    print("Recieved question: ", question)

    def stream() -> Generator:
        global run_id, trace_url, feedback_recorded

        q = Queue()
        job_done = object()

        llm = ChatOpenAI(model="gpt-3.5-turbo-16k", streaming=True, temperature=0, callbacks=[QueueCallback(q)])

        def task():
            agent = get_agent(llm, chat_history)
            agent.invoke({"input": question, "chat_history": chat_history}, config=runnable_config)
            q.put(job_done)

        t = Thread(target=task)
        t.start()

        content = ""

        while True:
            try:
                next_token = q.get(True, timeout=1)
                if next_token is job_done:
                    break
                content += next_token
                yield next_token
            except Empty:
                continue
            
        if not run_id and run_collector.traced_runs:
            run = run_collector.traced_runs[0]
            run_id = run.id
                
    return StreamingResponse(stream())

@app.post("/feedback")
async def send_feedback(request: Request):
    global run_id, feedback_recorded
    if feedback_recorded or run_id is None:
        return {"result": "Feedback already recorded or no chat session found", "code": 400}
    data = await request.json()
    score = data.get("score")
    client.create_feedback(run_id, "user_score", score=score)
    feedback_recorded = True
    return {"result": "posted feedback successfully", "code": 200}

trace_url = None
@app.post("/get_trace")
async def get_trace(request: Request):
    global run_id, trace_url
    if trace_url is None and run_id is not None:
        trace_url = client.share_run(run_id)
    if run_id is None:
        return {"result": "No chat session found", "code": 400}
    return trace_url if trace_url else {"result": "Trace URL not found", "code": 400}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
