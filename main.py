"""Main entrypoint for the app."""
import logging
import os
from typing import Literal, Optional, Union

import weaviate
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import HumanMessage, AIMessage
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableConfig, RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import Weaviate
from langsmith import Client
from operator import itemgetter

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

_PROVIDER_MAP = {
    "openai": ChatOpenAI,
    "anthropic": ChatAnthropic,
}

_MODEL_MAP = {
    "openai": "gpt-3.5-turbo",
    "anthropic": "claude-instant-v1-100k",
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

    _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone Question:"""

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    _template = """
    You are an expert programmer and problem-solver, tasked to answer any question about Langchain. Using the provided context, answer the user's question to the best of your ability using the resources provided.
    If you really don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
    Anything between the following markdown blocks is retrieved from a knowledge bank, not part of the conversation with the user.
    <context>
        {context}
    <context/>"""

    if chat_history:
        _inputs = RunnableMap(
            {
                "standalone_question": {
                    "question": lambda x: x["question"],
                    "chat_history": lambda x: x["chat_history"],
                }
                | CONDENSE_QUESTION_PROMPT
                | model
                | StrOutputParser(),
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
        )
        _context = {
            "context": itemgetter("standalone_question") | retriever,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _template),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        )
    else:
        _inputs = RunnableMap(
            {
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
        )
        _context = {
            "context": itemgetter("question") | retriever,
            "question": lambda x: x["question"],
            "chat_history": lambda x: x["chat_history"],
        }
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _template),
                ("human", "{question}"),
            ]
        )

    final_model = ChatOpenAI(model="gpt-3.5-turbo-16k") if model_provider == "openai" else ChatAnthropic(model_name="claude-2")
    chain = (
        _inputs
        | _context
        | prompt
        | final_model
        | StrOutputParser()
    )

    return chain


def _get_retriever():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]

    embeddings = OpenAIEmbeddings()
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=client,
        index_name="LangChain_newest_idx",
        text_key="text",
        embedding=embeddings,
        by_text=False,
        attributes=["source"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=10))

def _process_chat_history(chat_history):
    processed_chat_history = []
    for chat in chat_history:
        if "question" in chat:
            processed_chat_history.append(HumanMessage(content=chat.pop("question")))
        if "result" in chat:
            processed_chat_history.append(AIMessage(content=chat.pop("result")))
    return processed_chat_history


@app.post("/chat")
async def chat_endpoint(request: Request):
    global run_id, feedback_recorded, trace_url
    run_id = None
    trace_url = None
    feedback_recorded = False
    run_collector.traced_runs = []

    data = await request.json()
    question = data.get("message")
    model_type = data.get("model")
    chat_history = data.get("history", [])
    conversation_id = data.get("conversation_id")

    retriever = _get_retriever()
    chat_history = _process_chat_history(chat_history)

    # source_docs = retriever.invoke(question) # opportunity to return source documents
    # context = [doc.page_content for doc in source_docs]

    qa_chain = create_chain(
        retriever=retriever, model_provider=model_type, chat_history=chat_history
    )
    print("Received question: ", question)

    async def stream():
        global run_id, trace_url, feedback_recorded
        result = ""
        try:
            async for s in qa_chain.astream(
                {"question": question, "chat_history": chat_history},
                config={
                    **runnable_config,
                    "metadata": {"conversation_id": conversation_id},
                },
            ):
                print(s, end="", flush=True)
                result += s
                yield s
            if not run_id and run_collector.traced_runs:
                run = run_collector.traced_runs[0]
                run_id = run.id

        except Exception as e:
            logging.error(e)
            yield "Sorry, something went wrong. Try again." + "\n"

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
