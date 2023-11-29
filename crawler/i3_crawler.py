from bs4 import BeautifulSoup
import requests
import os
import firebase_admin
from firebase_admin import credentials, firestore

def i3_crawler(document_id: str, db):
    def extract_text_with_formatting(element):
        text_segments = []
        block_tags = {'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'blockquote'}
        for child in element.children:
            if not hasattr(child, 'attrs'):
                text_segments.append(child)
                continue
            if child.name in ['table', 'thead', 'tbody', 'tfoot', 'tr', 'td', 'th']:
                continue
            if 'class' in child.attrs and 'table' in child.attrs['class']:
                continue
            if child.name in block_tags:
                text_segments.append(child.get_text() + '\n')
            else:
                text_segments.append(child.get_text())
        return ''.join(text_segments)

    def extract_table_data(table):
        table_data = []
        for row in table.find_all('tr'):
            row_data = []
            for cell in row.find_all(['td', 'th']):
                row_data.append(cell.get_text(strip=True))
            table_data.append('\t'.join(row_data))
        return '\n'.join(table_data)

    def scrape_site(document_id):
        doc_ref = db.collection('files').document(document_id)
        doc = doc_ref.get()
        if not doc.exists:
            raise Exception(f"Document with ID {document_id} does not exist.")

        target_url = doc.to_dict().get('url')
        if not target_url:
            raise Exception(f"No URL found in the document with ID {document_id}.")

        login_url = "https://i3.vblh.de/login"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        with requests.Session() as session:
            response = session.get(login_url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            token_value = soup.find('input', {'name': '_token'})['value']
            
            username = os.environ.get('USERNAME')
            password = os.environ.get('PASSWORD')

            login_data = {
                'username': username,
                'password': password,
                '_token': token_value
            }
            post = session.post(login_url, data=login_data, headers=headers)
            
            if post.status_code != 200:
                raise Exception(f"Login failed with status code: {post.status_code}")

            login_error_indicators = ["Incorrect password", "Login failed", "Invalid credentials"]
            for indicator in login_error_indicators:
                if indicator in post.text:
                    raise Exception(f"Login failed: {indicator}")

            target_page = session.get(target_url, headers=headers)
            if target_page.status_code != 200:
                raise Exception(f"Received status code {target_page.status_code} when accessing target page.")

            soup = BeautifulSoup(target_page.content, 'html.parser')

            for placeholder in soup.find_all('span', class_='placeholder'):
                placeholder.decompose()

            scraped_text = ""

            entry_container = soup.find('div', class_='entry-container')
            if entry_container:
                scraped_text += extract_text_with_formatting(entry_container)
            else:
                scraped_text += "\n'entry-container' not found.\n"

            entry_descriptions = soup.find_all('div', class_='entry_description text-bigger-sm ck-content')
            for entry_description in entry_descriptions:
                scraped_text += extract_text_with_formatting(entry_description)

            table = soup.find('table')
            if table:
                scraped_text += "\n" + extract_table_data(table) + "\n\n"

            entry_concept_bodies = soup.find_all('div', class_='entry_concept_body ck-content')
            for entry_concept_body in entry_concept_bodies:
                scraped_text += extract_text_with_formatting(entry_concept_body)

            # Preparing the document for the FastAPI server upload
            doc_data = doc.to_dict()  # This is the data fetched from Firestore.
            BEARER_TOKEN = os.environ.get("BEARER_TOKEN")
            headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    
            document = {
                'id': document_id,
                'text': scraped_text,
                'metadata': {
                    'source': doc_data.get('source'),
                    'source_id': doc_data.get('source_id'),
                    'url': doc_data.get('url'),
                    'created_at': doc_data.get('created_at'),
                    'author': doc_data.get('author')
                }
            }
    
            endpoint_url = "https://chat-retrieval-api-avygm4cpgq-ey.a.run.app/upsert"
            response = requests.post(endpoint_url, headers=headers, json={"documents": [document]})
    
            # Check the response and maybe act upon it
            if response.status_code == 200:
                # Update Firestore document with the scraped text
                try:
                    doc_ref.update({'newText': scraped_text})
                except Exception as e:
                    raise Exception(f"Failed to update Firestore document for Document ID {document_id}. Error: {e}")
                return response.json()  # Returning the response from the API call
            else:
                raise Exception(f"Failed to upload data for Document ID {document_id}. Status: {response.status_code}, Response: {response.json()}")


    try:
        return scrape_site(document_id)
    except Exception as e:
        raise e
