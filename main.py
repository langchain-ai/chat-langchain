"""Main entrypoint for the app."""
import os
from operator import itemgetter
from typing import AsyncIterator, Dict, List, Optional
import json
import weaviate
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.tracers.log_stream import RunLogPatch
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableMap
from langchain.vectorstores import Weaviate
from langsmith import Client
from pydantic import BaseModel

from constants import WEAVIATE_DOCS_INDEX_NAME


RESPONSE_TEMPLATE = """You are an expert programmer and problem-solver, tasked to answer any question about Langchain. Using the provided context, answer the user's question to the best of your ability using the resources provided.
Generate a comprehensive and informative answer (but no more than 80 words) for a given question based solely on the provided search results (URL and content). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Do not repeat text. Cite search results using [${{number}}] notation. Only cite the most relevant results that answer the question accurately. Place these citations at the end of the sentence or paragraph that reference them - do not put them all at the end. If different results refer to different entities within the same name, write separate answers for each entity.
If there is nothing in the context relevant to the question at hand, just say "Hmm, I'm not sure." Don't try to make up an answer.
Anything between the following `context`  html blocks is retrieved from a knowledge bank, not part of the conversation with the user. 
<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm not sure." Don't try to make up an answer. Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, not part of the conversation with the user."""

REPHRASE_TEMPLATE = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


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
    return weaviate_client.as_retriever(search_kwargs=dict(k=6))


def create_retriever_chain(chat_history, llm, retriever: BaseRetriever):
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)

    if chat_history:
        condense_question_chain = (
            {
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
            }
            | CONDENSE_QUESTION_PROMPT
            | llm
            | StrOutputParser()
        ).with_config(
            {
                "run_name": "condense_question",
            }
        )
        retriever_chain = condense_question_chain | retriever
    else:
        retriever_chain = (itemgetter("question")) | retriever
    return retriever_chain


def format_docs(docs, max_tokens=200):
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def create_chain(
    llm,
    retriever_chain,
) -> Runnable:
    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
    ).with_config(run_name="RetrieveDocs")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return _context | response_synthesizer


async def transform_stream_for_client(
    stream: AsyncIterator[RunLogPatch],
) -> AsyncIterator[str]:
    async for chunk in stream:
        for op in chunk.ops:
            all_sources = []
            if op["path"] == "/logs/0/final_output":
                # Send source urls when they become available
                url_set = set()
                for doc in op["value"]["output"]:
                    if doc.metadata["source"] in url_set:
                        continue

                    url_set.add(doc.metadata["source"])
                    all_sources.append(
                        doc.metadata["title"] + ":" + doc.metadata["source"]
                    )
                if all_sources:
                    src = {"sources": "\n".join(all_sources)}
                    yield f"{json.dumps(src)}\n"

            elif op["path"] == "/streamed_output/-":
                # Send stream output
                yield f'{json.dumps({"tok": op["value"]})}\n'

            elif not op["path"] and op["op"] == "replace":
for                # Send final output
                yield f'{json.dumps({"run_id": str(op["value"]["id"])})}\n'


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

    llm = ChatOpenAI(
        model="gpt-3.5-turbo-16k",
        streaming=True,
        temperature=0,
    )
    docs_chain = create_retriever_chain(converted_chat_history, llm, get_retriever())
    answer_chain = create_chain(llm, docs_chain.with_config(run_name="FindDocs"))
    stream = answer_chain.astream_log(
        {
            "question": question,
            "chat_history": converted_chat_history,
            "converted_chat_history": converted_chat_history,
        },
        config={"metadata": metadata},
        include_names=["FindDocs"],
        include_tags=["FindDocs"],
    )
    return StreamingResponse(transform_stream_for_client(stream))


@app.post("/feedback")
async def send_feedback(request: Request):
    data = await request.json()
    score = data.get("score")
    run_id = data.get("run_id")  # TODO：prevent duplicate feedback
    if run_id is None:
        return {
            "result": "Feedback already recorded or no chat session found",
            "code": 400,
        }

    client.create_feedback(run_id, "user_score", score=score)
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
