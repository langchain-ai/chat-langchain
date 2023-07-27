"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import platform

from dotenv import load_dotenv
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

load_dotenv()


def ingest_docs():
    """Get documents from web pages."""
    if platform.system() == "Windows":
        loader = ReadTheDocsLoader("api.python.langchain.com/en/latest/", "utf-8-sig")
    else:
        loader = ReadTheDocsLoader("api.python.langchain.com/en/latest/")

    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
