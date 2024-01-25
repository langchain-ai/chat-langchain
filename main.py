"""Main entrypoint for the app."""
import os

from langchain.globals import set_debug
import asyncio
from typing import Dict, List, Optional, Union
from uuid import UUID

from chromadb.utils import embedding_functions

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import langsmith
from langsmith import Client
from langserve import add_routes

from pydantic import BaseModel

# from croptalk.model_llm import model
from croptalk.model_openai_functions import model

from dotenv import load_dotenv
load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')

set_debug(True)


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


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


model_name = os.getenv("MODEL_NAME")
vectorestore_dir = os.getenv("VECTORSTORE_DIR")
collection_name = os.getenv("VECTORSTORE_COLLECTION")
emb_fn = embedding_functions.DefaultEmbeddingFunction()
model = ModelFactory(
    llm_model_name=model_name,
    vectorestore_dir=vectorestore_dir,
    collection_name=collection_name,
    embedding_function=emb_fn,
).get_model()
add_routes(app, model, path="/chat",
           input_type=ChatRequest, config_keys=["metadata"])


class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


@app.post("/feedback")
async def send_feedback(body: SendFeedbackBody):
    client.create_feedback(
        body.run_id,
        body.key,
        score=body.score,
        comment=body.comment,
        feedback_id=body.feedback_id,
    )
    return {"result": "posted feedback successfully", "code": 200}


class UpdateFeedbackBody(BaseModel):
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None


@app.patch("/feedback")
async def update_feedback(body: UpdateFeedbackBody):
    feedback_id = body.feedback_id
    if feedback_id is None:
        return {
            "result": "No feedback ID provided",
            "code": 400,
        }
    client.update_feedback(
        feedback_id,
        score=body.score,
        comment=body.comment,
    )
    return {"result": "patched feedback successfully", "code": 200}


# TODO: Update when async API is available
async def _arun(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, func, *args, **kwargs)


async def aget_trace_url(run_id: str) -> str:
    for i in range(5):
        try:
            await _arun(client.read_run, run_id)
            break
        except langsmith.utils.LangSmithError:
            await asyncio.sleep(1**i)

    if await _arun(client.run_is_shared, run_id):
        return await _arun(client.read_run_shared_link, run_id)
    return await _arun(client.share_run, run_id)


class GetTraceBody(BaseModel):
    run_id: UUID


@app.post("/get_trace")
async def get_trace(body: GetTraceBody):
    run_id = body.run_id
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(str(run_id))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
