"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import fnmatch
import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import PyPDFLoader

def ingest_docs(file):
    documents = []
   
    loader = PyPDFLoader('./docs/'+file)
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings(openai_api_key="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")
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

if __name__ == "__main__":
    ingest_docs()
