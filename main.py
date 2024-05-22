"""Main entrypoint for the app."""

import asyncio
import logging
from typing import Dict, List, Optional, Union
from uuid import UUID

import langsmith
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain.globals import set_debug
from langserve import add_routes
from langsmith import Client
from pydantic.v1 import BaseModel
from typing import Dict, Any


from croptalk.model_openai_functions import model, memory

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


add_routes(app, model, path="/chat",
           input_type=ChatRequest, config_keys=["metadata"])


@app.post("/clear_memory")
async def clear_memory():
    logging.info("CLEARING MEMORY")
    try:
        memory.clear()
        code = 200
        result = "success"

    except Exception as e:
        logging.info(f"Exception from clearing memory : {e}")
        code = 500
        result = e

    return {"result": result, "code": code}


class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


@app.post("/feedback")
async def send_feedback(body: Dict[Any, Any]):

    # instantiate SendFeedbackBody for type checking
    send_feedback_body = SendFeedbackBody(**body)

    # Call create_feedback method from langsmith client
    client.create_feedback(
        run_id=send_feedback_body.run_id,
        key=send_feedback_body.key,
        score=send_feedback_body.score,
        comment=send_feedback_body.comment,
        feedback_id=send_feedback_body.feedback_id
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
            await asyncio.sleep(1 ** i)

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
