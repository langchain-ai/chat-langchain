"""Load html from files, clean up, split, ingest into Weaviate."""
from bs4 import BeautifulSoup as Soup
import weaviate
import os

from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate

WEAVIATE_URL=os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY=os.environ["WEAVIATE_API_KEY"]

def ingest_docs():
    """Get documents from web pages."""

    urls = [
        "https://api.python.langchain.com/en/latest/api_reference.html#module-langchain",
        "https://python.langchain.com/docs/get_started", 
        "https://python.langchain.com/docs/modules", 
        "https://python.langchain.com/docs/guides",
        "https://python.langchain.com/docs/ecosystem",
        "https://python.langchain.com/docs/additional_resources",
    ]

    documents = []
    for url in urls:
        loader = RecursiveUrlLoader(url=url, max_depth=1 if url == urls[0] else 3, extractor=lambda x: Soup(x, "lxml").text, prevent_outside=False)
        documents += loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    print('before splitting', len(documents))
    docs = text_splitter.split_documents(documents)
    print('after splitting', len(docs))

    embeddings = OpenAIEmbeddings(chunk_size=200) # rate limit

    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))
    client.schema.delete_all()
    print(client.schema.get()) # should be empty

    batch_size = 100 # to handle batch size limit 
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        Weaviate.from_documents(batch, embeddings, client=client, by_text=False, index_name="LangChain_idx")

    print("LangChain now has this many vectors", client.query.aggregate("LangChain_idx").with_meta_count().do())
    
if __name__ == "__main__":
    ingest_docs()
    