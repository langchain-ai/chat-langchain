"""Load html from files, clean up, split, ingest into Weaviate."""

import os

from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from loguru import logger


def ingest_docs():
    """Get documents from web pages."""
    loaders = []

    namespace = "vic-policies"

    doc_folder = f"data/{namespace}/"
    # doc_folder = "data/security-docs/"
    # doc_folder = "data/prophet-docs/"

    for file in os.listdir(doc_folder):
        if not file.startswith("."):
            filename = f"{doc_folder}{file}"
            logger.info(f"Loading {filename}")
            loaders.append(PyMuPDFLoader(filename))

    docs = []
    for loader in loaders:
        try:
            logger.debug(f"Loading {loader.file_path}")
            docs.extend(loader.load())
        except Exception as e:
            logger.error(f"Failed to load {loader}: {e}")

    # loader = PyMuPDFLoader("data/security-docs/2022 Montoux Solution Overview.pdf")
    text_splitter = RecursiveCharacterTextSplitter()

    documents = text_splitter.split_documents(docs)
    # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    embeddings = OpenAIEmbeddings()

    PGVector.from_documents(
        embedding=embeddings,
        documents=documents,
        collection_name=namespace,
        pre_delete_collection=True,
    )


if __name__ == "__main__":
    load_dotenv()
    ingest_docs()
