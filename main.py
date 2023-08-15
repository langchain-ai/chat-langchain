"""Main entrypoint for the app."""
import asyncio
import logging
import os
from typing import Literal, Optional, Union

import weaviate
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic, ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import Runnable, RunnableConfig
from langchain.vectorstores import Weaviate
from langsmith import Client

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

    _template = """You are an expert programmer, tasked to answer any question about Langchain. Be as helpful as possible. 
    
    Anything between the following markdown blocks is retrieved from a knowledge bank, not part of the conversation with the user. 
    <context>
        {context} 
    <context/>
                    
    Conversation History:               
    <conversation_history>
    {history}
    <conversation_history/>
    
    Answer the user's question to the best of your ability: {question}
    Helpful Answer:"""

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"], template=_template
    )
    memory = ConversationBufferMemory(input_key="question", memory_key="history")
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


@app.post("/chat")
async def chat_endpoint(request: Request):
    global run_id, access_counter
    run_id = None
    access_counter = 0
    run_collector.traced_runs = []

    data = await request.json()
    question = data.get("message")
    model_type = data.get("model")
    chat_history = data.get("history", [])

    retriever = _get_retriever()
    qa_chain = create_chain(
        retriever=retriever, model_provider=model_type, chat_history=chat_history
    )
    print("Recieved question: ", question)

    async def stream():
        result = ""
        try:
            for s in qa_chain.stream(question, config=runnable_config):
                result += s["result"]
                yield s["result"]
                await asyncio.sleep(0)

        except Exception as e:
            logging.error(e)
            yield "Sorry, something went wrong. Try again." + "\n"

    return StreamingResponse(stream())


access_counter = 0


@app.post("/feedback")
async def send_feedback(request: Request):
    global run_id, access_counter
    data = await request.json()
    score = data.get("score")
    if not run_id:
        run = run_collector.traced_runs[0]
        run_id = run.id
        access_counter += 1
        if access_counter >= 2:
            run_collector.traced_runs = []
            access_counter = 0
    client.create_feedback(run_id, "user_score", score=score)
    return {"result": "posted feedback successfully", "code": 200}


@app.post("/get_trace")
async def get_trace(request: Request):
    global run_id, access_counter
    if run_id is None:
        run = run_collector.traced_runs[0]
        run_id = run.id
        access_counter += 1
        if access_counter >= 2:
            run_collector.traced_runs = []
            access_counter = 0
    url = client.share_run(run_id)
    return url


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
