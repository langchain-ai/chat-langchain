"""Main entrypoint for the app."""
import logging
import os
from collections.abc import Generator
from operator import itemgetter
from queue import Empty, Queue
from threading import Thread
from typing import Dict, Generator, List, Optional

import weaviate
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.manager import collect_runs, trace_as_chain_group
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable
from langchain.vectorstores import Weaviate
from langsmith import Client
from pydantic import BaseModel

from constants import WEAVIATE_DOCS_INDEX_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
}

_MODEL_MAP = {
    "openai": "gpt-3.5-turbo-16k",
    "anthropic": "claude-2",
}

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

run_id = None
feedback_recorded = False

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


def get_retriever():
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=OpenAIEmbeddings(chunk_size=200),
        by_text=False,
        attributes=["source", "title"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=3))


def create_retriever_chain(chat_history, llm, retriever: BaseRetriever):
    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone Question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    chain_meta = {
        "run_name": "retrieve_docs",
        "metadata": {
            "subchain": "retriever_chain",
        },
    }

    if chat_history:
        retriever_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser()
            | retriever
        ).with_config(chain_meta)
    else:
        retriever_chain = ((itemgetter("question")) | retriever).with_config(chain_meta)
    return retriever_chain


def format_docs(docs):
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_chain(
    llm,
    retriever_chain,
) -> Runnable:
    _template = """
    You are an expert programmer and problem-solver, tasked to answer any question about Langchain. Using the provided context, answer the user's question to the best of your ability using the resources provided.
    If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.
    Anything between the following `context`  html blocks is retrieved from a knowledge bank, not part of the conversation with the user. 
    <context>
        {context} 
    <context/>

    REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user.
    """

    _context = {
        "context": retriever_chain | format_docs,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        {
            "run_name": "synthesize_response",
            "metadata": {
                "subchain": "response_synthesizer",
            },
        }
    )
    chain = _context | response_synthesizer
    return chain


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    # https://gist.github.com/mortymike/70711b028311681e5f3c6511031d5d43

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs: any) -> None:
        return self.q.empty()


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]]
    conversation_id: Optional[str]


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global run_id, feedback_recorded, trace_url
    run_id = None
    trace_url = None
    feedback_recorded = False

    question = request.message
    chat_history = request.history or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    metadata = {
        "conversation_id": request.conversation_id,
    }

    def stream() -> Generator:
        global run_id, trace_url, feedback_recorded

        q = Queue()
        job_done = object()

        llm = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            streaming=True,
            temperature=0,
            callbacks=[QueueCallback(q)],
        )

        llm_without_callback = ChatOpenAI(
            model="gpt-3.5-turbo-16k",
            streaming=True,
            temperature=0,
        )

        def task():
            global run_id
            with collect_runs() as run_collector:
                with trace_as_chain_group(
                    "chat-langchain",
                    inputs={"question": question, "chat_history": chat_history},
                ) as group_manager:
                    retriever_chain = create_retriever_chain(
                        chat_history, llm_without_callback, get_retriever()
                    )
                    chain = create_chain(llm, retriever_chain)
                    docs = retriever_chain.invoke(
                        {"question": question, "chat_history": chat_history},
                        config={"metadata": metadata, "callbacks": group_manager},
                    )
                    url_set = set()
                    for doc in docs:
                        if doc.metadata["source"] in url_set:
                            continue
                        q.put(
                            doc.metadata["title"] + ":" + doc.metadata["source"] + "\n"
                        )
                        url_set.add(doc.metadata["source"])
                    if len(docs) > 0:
                        q.put("SOURCES:----------------------------")
                    res = chain.invoke(
                        {
                            "question": question,
                            "chat_history": converted_chat_history,
                            "context": docs,
                        },
                        config={"metadata": metadata, "callbacks": group_manager},
                    )
                    q.put(job_done)
                    group_manager.on_chain_end({"output": res})
                # The run ID for the grouped trace
                run_id = run_collector.traced_runs[0].id

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

    return StreamingResponse(stream())


@app.post("/feedback")
async def send_feedback(request: Request):
    global run_id, feedback_recorded
    if feedback_recorded or run_id is None:
        return {
            "result": "Feedback already recorded or no chat session found",
            "code": 400,
        }
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
