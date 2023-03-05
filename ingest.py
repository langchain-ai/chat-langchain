"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv

load_dotenv()


def ingest_docs():
    loader = ReadTheDocsLoader("aide.blank.app/")
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    # limit to 10 documents for now to avoid costs
    documents = text_splitter.split_documents(raw_documents)[:10]
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)



if __name__ == "__main__":
    ingest_docs()
