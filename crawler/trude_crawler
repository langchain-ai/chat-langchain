import requests
from PyPDF2 import PdfReader
import io
import re
import os
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase Admin SDK
cred = credentials.ApplicationDefault()
firebase_admin.initialize_app(cred, {
    'project_id': os.environ.get('GCP_PROJECT')
})
db = firestore.client()

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

def trude_crawler(document_id: str):
    doc_ref = db.collection('files').document(document_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise Exception(f"Document with ID {document_id} does not exist.")

    target_url = doc.to_dict().get('url')
    if not target_url:
        raise Exception(f"No URL found in the document with ID {document_id}.")

    # Initialize session and make initial request if START_URL is set
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

    endpoint_url = "https://chat-retrieval-api-avygm4cpgq-ey.a.run.app/upsert"
    api_response = requests.post(endpoint_url, headers=headers, json={"documents": [document]})

    if api_response.status_code == 200:
        try:
            doc_ref.update({'newText': extracted_text})
        except Exception as e:
            raise Exception(f"Error updating Firestore document: {e}")
        return api_response.json()  # Returning the response from the API call
    else:
        raise Exception(f"Failed to upload data. Status: {api_response.status_code}, Response: {api_response.json()}")

