"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.document_loaders import RecursiveUrlLoader
import numpy as np

def ingest_docs():
    """Get documents from web pages."""
    loader = ReadTheDocsLoader("api.python.langchain.com/en/latest", features="lxml")
    loader2 = ReadTheDocsLoader("python.langchain.com", features="lxml")
    # loader2 = RecursiveUrlLoader(url="https://python.langchain.com/docs/get_started/introduction", max_depth=2, extractor=lambda x: Soup(x, "lxml").text)

    raw_documents = loader.load()
    print(len(raw_documents))
    raw_documents += loader2.load()
    print(len(raw_documents))
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    print(len(documents))
    embeddings = OpenAIEmbeddings(chunk_size=100)
    print(embeddings)
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    # Ensure all embeddings have the same length
    max_length = max(len(e) for e in embeddings)
    padded_embeddings = np.array([e + tuple([0]*(max_length-len(e))) for e in embeddings])
    
    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    ingest_docs()
    