from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_voyageai import VoyageAIEmbeddings

def get_embeddings_model() -> Embeddings:
    # return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)
    # return VoyageAIEmbeddings(model="voyage-3-lite")
    return VoyageAIEmbeddings(model="voyage-law-2")
