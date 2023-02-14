"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
from pathlib import Path

from bs4 import BeautifulSoup
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def clean_data(data):
    soup = BeautifulSoup(data, features="lxml")
    text = soup.find_all("main", {"id": "main-content"})[0].get_text()
    return "\n".join([t for t in text.split("\n") if t])


def ingest_docs():
    """Get documents from web pages."""
    raw_documents = []
    metadatas = []
    for p in Path("langchain.readthedocs.io/en/latest/").rglob("*"):
        if p.is_dir():
            continue
        with open(p) as f:
            raw_documents.append(clean_data(f.read()))
            metadatas.append({"source": p})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.create_documents(raw_documents, metadatas=metadatas)

    # Load Data to vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
