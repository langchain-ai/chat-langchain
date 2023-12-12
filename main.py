"""Main entrypoint for the app."""
import asyncio
import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence, Union
from uuid import UUID

import langsmith
import weaviate
from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from langserve import add_routes
from langsmith import Client
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, firestore

from crawler.i3_crawler import i3_crawler
from crawler.trude_crawler import trude_crawler
from datastore.factory import get_datastore
from services.file import get_document_from_file
from chains.rag_chain import create_answer_chain
from models.models import Document

from models.api import (
    DeleteRequest,
    DeleteResponse,
    UpsertResponse,
)

class CrawlerRequest(BaseModel):
    document_id: str
  
class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]
  
class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        api_key_header = request.headers.get("X-API-Key")
        expected_api_key = os.getenv("CHAT_API_KEY")  # Your API Key stored in an environment variable

        if not api_key_header or api_key_header != expected_api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API Key")

        response = await call_next(request)
        return response

class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None

class UpdateFeedbackBody(BaseModel):
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None

class GetTraceBody(BaseModel):
    run_id: UUID

datastore = None

client = Client()

# Initialize Firebase Admin SDK
cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred, {
    'project_id': os.environ.get('GCP_PROJECT')
})

# Make the Firestore client available globally
db = firestore.client()

app = FastAPI()
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

answer_chain = create_answer_chain()

add_routes(app, answer_chain, path="/chat", input_type=ChatRequest)


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


@app.post("/get_trace")
async def get_trace(body: GetTraceBody):
    run_id = body.run_id
    if run_id is None:
        return {
            "result": "No LangSmith run ID provided",
            "code": 400,
        }
    return await aget_trace_url(str(run_id))

# Upsert contents from a pdf file
@app.post(
    "/upsert-file",
    response_model=UpsertResponse,
)
async def upsert_file(
    file: UploadFile = File(...),
    metadata: Optional[str] = Form(None),
):
    try:
        metadata_obj = (
            Document.parse_raw(metadata)
        )
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"str({e})")

    document = await get_document_from_file(file, metadata_obj)

    try:
        ids = await datastore.upsert([document])
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail=f"str({e})")


# Upsert a single document by providing the document id
@app.post("/upsert")
async def upsert_document(request: CrawlerRequest):
    try:
        # Fetch the document from Firestore
        doc_ref = db.collection("files").document(request.document_id)
        doc = doc_ref.get()
        if not doc.exists:
            return {"result": "Document not found", "code": 404}

        # Get the URL from the document
        url = doc.to_dict().get("url")
        if url:
            # Check if the URL matches a specific pattern and call the appropriate crawler
            if "i3.vblh" in url:
                response = await i3_crawler(request.document_id, db, datastore)
            elif "onlinetrude" in url:
                response = await trude_crawler(request.document_id, db, datastore)
            else:
                return {"result": "URL pattern not recognized", "code": 400}
            return response
        else:
            return {"result": "URL not found in document", "code": 400}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Refresh all the documents in Firebase
@app.post("/refresh")
async def refresh_documents():
    try:
        # Fetch all documents from the "files" collection
        docs = db.collection("files").stream()

        # Iterate over each document and call upsert_document
        results = []
        for doc in docs:
            document_id = doc.id
            crawler_request = CrawlerRequest(document_id=document_id)
            response = await upsert_document(crawler_request)
            results.append({
                "document_id": document_id,
                "response": response
            })

        return {"results": results, "code": 200}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete(
    "/delete",
    response_model=DeleteResponse,
)
async def delete(
    request: DeleteRequest = Body(...),
):
    if not (request.ids or request.filter or request.delete_all):
        raise HTTPException(
            status_code=400,
            detail="One of ids, filter, or delete_all is required",
        )
    try:
        success = await datastore.delete(
            ids=request.ids,
            filter=request.filter,
            delete_all=request.delete_all,
        )
        return DeleteResponse(success=success)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")
        

@app.on_event("startup")
async def startup():
    global datastore
    datastore = await get_datastore()
    print("Datastore initialized:", datastore) 


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
