"""Load html from files, clean up, split, ingest into Weaviate."""
import os
import pickle

from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from loguru import logger


def ingest_docs():
    """Get documents from web pages."""
    loaders = []

    # doc_folder = "data/accounting-docs/"
    doc_folder = "data/security-docs/"
    # doc_folder = "data/prophet-docs/"

    for file in os.listdir(doc_folder):
        if not file.startswith("."):
            filename = f"{doc_folder}{file}"
            logger.info(f"Loading {filename}")
            loaders.append(PyMuPDFLoader(filename))

    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    # loader = PyMuPDFLoader("data/security-docs/2022 Montoux Solution Overview.pdf")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    documents = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    load_dotenv()
    ingest_docs()
