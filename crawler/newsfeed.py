import openai
import os
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import HTTPException

# Set OpenAI API key
openai.api_type = "azure"
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')
openai.deployment_id = "gpt-4"

def summarize_text(text, source_text):
    prompt = f"Fasse kurz zusammen, worum es in folgendem Text geht. Halte dich kurz und nutze maximal 3 Sätze und nicht mehr als 50 Wörter. Beginne deinen ersten Satz direkt mit dem Objekt, um das es geht.:\n\n{text}"
    completion = openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": "Du bist ein Assistent der Texte auf Deutsch zusammenfasst und Mitarbeiter über Neuerungen informiert."},
            {"role": "user", "content": prompt}
        ]
    )
    summary = completion.choices[0].message.content
    return f"{summary} Quelle: {source_text}"

def compare_and_summarize_texts(new_text, old_text, source_text):
    prompt = f"Vergleiche folgende zwei Texte. Fasse zusammen, was es Neues gibt. Halte dich kurz und beziehe dich nur auf die Neuerung. Beginne deinen ersten Satz direkt mit dem Objekt, um das es geht.:\n\nAlt:\n{old_text}\n\nNeu:\n{new_text}"
    completion = openai.ChatCompletion.create(
        messages=[
            {"role": "system", "content": "Du bist ein Assistent der Mitarbeiter über Neuerungen informiert. Dafür vergleichst du Informationen im Intranet und informierst über Änderungen."},
            {"role": "user", "content": prompt}
        ]
    )
    comparison_summary = completion.choices[0].message.content
    return f"{comparison_summary} Quelle: {source_text}"

def update_all_users_with_summary(summary):
    users_ref = firestore.client().collection('users')
    docs = users_ref.stream()

    for doc in docs:
        user_data = doc.to_dict()
        current_news_summary = user_data.get('newsSummary', [])
        current_news_summary.append(summary)
        users_ref.document(doc.id).update({'newsSummary': current_news_summary})

async def create_newsfeed(document_id: str):
    try:
        print(f"Processing newsfeed creation for document ID: {document_id}")
        doc_ref = firestore.client().collection('files').document(document_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            print(f"Document with ID {document_id} not found")
            raise HTTPException(status_code=404, detail="Document not found")

        new_data = doc.to_dict()
        print(f"Document data fetched for ID {document_id}: {new_data}")

        if 'newText' not in new_data:
            print("newText field is missing in the document")
            return {"message": "newText not present in the document."}

        newText = new_data['newText']
        sourceText = new_data['source']
        oldText = new_data.get('oldText', '')

        if newText == oldText:
            print("No changes detected between newText and oldText")
            return {"message": "No changes detected. newText is identical to oldText."}

        if not oldText:
            print("Generating summary for newText")
            summary = summarize_text(newText, sourceText)
            update_all_users_with_summary(summary)
            result = {"summary": summary}
        else:
            print("Comparing newText and oldText and generating summary")
            comparison_summary = compare_and_summarize_texts(newText, oldText, sourceText)
            update_all_users_with_summary(comparison_summary)
            result = {"comparison_summary": comparison_summary}

        print(f"Updating Firestore document for ID {document_id} - setting oldText to newText")
        doc_ref.update({'oldText': newText})

        return result

    except Exception as e:
        print(f"Error in create_newsfeed for document ID {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
