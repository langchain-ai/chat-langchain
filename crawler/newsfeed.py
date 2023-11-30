import openai
import os
import firebase_admin
from firebase_admin import credentials, firestore
from fastapi import HTTPException

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def summarize_text(text, source_text):
    prompt = f"Fasse kurz zusammen, worum es in folgendem Text geht. Halte dich kurz und nutze maximal 3 Sätze und nicht mehr als 50 Wörter. Beginne deinen ersten Satz direkt mit dem Objekt, um das es geht.:\n\n{text}"
    completion = openai.Completion.create(
        model="gpt-4-1106-preview",
        prompt=prompt,
        max_tokens=100
    )
    summary = completion.choices[0].text.strip()
    return f"{summary} Quelle: {source_text}"

def compare_and_summarize_texts(new_text, old_text, source_text):
    prompt = f"Vergleiche folgende zwei Texte. Fasse zusammen, was es Neues gibt. Halte dich kurz und beziehe dich nur auf die Neuerung. Beginne deinen ersten Satz direkt mit dem Objekt, um das es geht.:\n\nAlt:\n{old_text}\n\nNeu:\n{new_text}"
    completion = openai.Completion.create(
        model="gpt-4-1106-preview",
        prompt=prompt,
        max_tokens=100
    )
    comparison_summary = completion.choices[0].text.strip()
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
        doc_ref = firestore.client().collection('files').document(document_id)
        doc = doc_ref.get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Document not found")

        new_data = doc.to_dict()

        if 'newText' not in new_data:
            return {"message": "newText not present in the document."}

        newText = new_data['newText']
        sourceText = new_data['source']
        oldText = new_data.get('oldText', '')
      
        # Check if newText and oldText are identical
        if newText == oldText:
            return {"message": "No changes detected. newText is identical to oldText."}
          
        # Perform the summarization or comparison
        if not oldText:
            summary = summarize_text(newText, sourceText)
            update_all_users_with_summary(summary)
            result = {"summary": summary}
        else:
            comparison_summary = compare_and_summarize_texts(newText, oldText, sourceText)
            update_all_users_with_summary(comparison_summary)
            result = {"comparison_summary": comparison_summary}

        # Update Firestore document - set oldText to newText
        doc_ref.update({'oldText': newText})

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
