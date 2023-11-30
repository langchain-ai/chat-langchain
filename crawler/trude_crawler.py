import requests
from PyPDF2 import PdfReader
import io
import re
import os
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from models.api import (
    DeleteRequest,
    DeleteResponse,
    QueryRequest,
    QueryResponse,
    UpsertRequest,
    UpsertResponse,
    AgentRequest,
)

from models.models import DocumentMetadata

async def upsert(datastore, request: UpsertRequest = Body(...)):
    try:
        print("Trying to use upsert function with datastore:", datastore)
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

def convert_to_pdf_url(target_url: str) -> str:
    match = re.search(r'/incident/(\d+)/?', target_url)
    if match:
        incident_id = match.group(1)
        return f'https://onlinetrude.genolotse.de/incident/pdf/create/{incident_id}'
    else:
        return "Invalid URL format"

def extract_text_from_pdf_stream(pdf_stream: io.BytesIO) -> str:
    try:
        reader = PdfReader(pdf_stream)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text() is not None])
    except Exception as e:
        return f"Error processing file: {e}"

async def trude_crawler(document_id: str, db, datastore):
    doc_ref = db.collection('files').document(document_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise Exception(f"Document with ID {document_id} does not exist.")

    target_url = doc.to_dict().get('url')
    if not target_url:
        raise Exception(f"No URL found in the document with ID {document_id}.")

    # Initialize session and make initial request if START_URL is set
    # Note: Consider using aiohttp for asynchronous HTTP requests
    session = requests.Session()
    start_url = os.environ.get('START_URL')
    if start_url:
        session.get(start_url)

    pdf_url = convert_to_pdf_url(target_url)
    if pdf_url == "Invalid URL format":
        raise Exception("Invalid target URL format.")

    response = session.get(pdf_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download PDF. Status code: {response.status_code}")

    pdf_stream = io.BytesIO(response.content)
    extracted_text = extract_text_from_pdf_stream(pdf_stream)

    # API call logic
    doc_data = doc.to_dict()
    BEARER_TOKEN = os.environ.get("BEARER_TOKEN") or "default_bearer_token"
    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN}"
    }

    document = {
        'id': document_id,
        'text': extracted_text,
        'metadata': {
            'source': doc_data.get('source'),
            'source_id': doc_data.get('source_id'),
            'url': doc_data.get('url'),
            'created_at': doc_data.get('created_at'),
            'author': doc_data.get('author')
        }
    }

    upsert_request = UpsertRequest(documents=[document])
    
    try:
        print("Trying to upsert with datastore:", datastore) 
        upsert_response = await upsert(datastore, upsert_request)
        # Update Firestore document with the scraped text
        try:
            doc_ref.update({'newText': extracted_text})
        except Exception as e:
            raise Exception(f"Failed to update Firestore document for Document ID {document_id}. Error: {e}")
        return upsert_response  # Returning the response from the upsert function
    except Exception as e:
        raise Exception(f"Failed to upsert data for Document ID {document_id}. Error: {e}")

