import requests
import os
import firebase_admin
import openai
from fastapi import FastAPI, File, Form, HTTPException, Depends, Body, UploadFile
from firebase_admin import credentials, firestore

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

# Set OpenAI API configurations for Azure
openai.api_type = "azure"
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_version = os.getenv('OPENAI_API_VERSION')

secure = os.getenv('EASYDAY_SECURE')
apiKey = os.getenv('EASYDAY_API_KEY')
secureUrl = os.getenv('EASYDAY_SECURE_URL')

async def upsert(datastore, request: UpsertRequest = Body(...)):
    try:
        print("Trying to use upsert function with datastore:", datastore)
        ids = await datastore.upsert(request.documents)
        return UpsertResponse(ids=ids)
    except Exception as e:
        print("Error:", e)
        raise HTTPException(status_code=500, detail="Internal Service Error")

async def summarize_text(text):
    prompt = f"Fasse das folgende Transkript ausführlich zusammen. Gliedere deine Zusammenfassung in sinnvolle Abschnitte, die jeweils eine Überschrift enthalten. Nutze deinen inneren Monolog, um deine Zusammenfassung zu prüfen. Es dürfen keine wichtigen Informationen verloren gehen.:\n\nStart des Transkripts:{text}"
    try:
        completion = openai.ChatCompletion.create(
            deployment_id="gpt-4",
            messages=[
                {"role": "system", "content": "Du bist ein Assistent der Transkripte ausführlich auf Deutsch zusammenfasst und in sinnvolle Abschnitte gliedert."},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error during summarization: {type(e).__name__}: {e}")
        return ""

async def process_easyday_data(db, datastore):
    url = "https://app.easyday.coach/methods/api.blocks.getBlocksTranscripts"
    data = {
        "secure": secure,
        "apiKey": apiKey
    }
    
    processed_ids = []
    
    response = requests.post(url, json=data)

    if response.status_code == 200:
        json_data = response.json()

        for item in json_data[:2]:  # Limit to first two elements for testing
            transcript = item["transcription"]["Transcript"]
            
            # Get the existing document from Firestore
            doc_ref = db.collection('easyday').document(item["_id"])
            doc = doc_ref.get()

            # Check if the document exists and if the transcript is identical
            if doc.exists and doc.to_dict().get('transcript') == transcript:
                print(f"Skipping {item['_id']} as the transcript is identical.")
                continue

            # If the transcript is new or different, process it
            summary = await summarize_text(transcript)
            doc_ref.set({
                'title': item["title"],
                'id': item["_id"],
                'url': f"https://app.easyday.coach/blockviewcopilot/{item['_id']}/{secureUrl}/{apiKey}",
                'transcript': transcript,
                'summary': summary
            })

            # Prepare document to upsert to vector store
            document = {
                'id': item["_id"],
                'text': summary,
                'title': item["title"],
                'source': f"https://app.easyday.coach/blockviewcopilot/{item['_id']}/{secureUrl}/{apiKey}"
            }
            
            upsert_request = UpsertRequest(documents=[document])

            # Call the upsert function      
            try:
                print("Trying to upsert with datastore:", datastore) 
                upsert_response = await upsert(datastore, upsert_request)
                
                # Return the upsert_response
                # If processed, add the ID to the list
                processed_ids.append(item["_id"])
            
            except Exception as e:
                raise Exception(f"Failed to upsert data for Document ID {document_id}. Error: {e}")     
            
    else:
        print("Error:", response.status_code)

    return processed_ids
