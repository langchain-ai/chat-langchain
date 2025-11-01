import os
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from backend.constants import OLLAMA_BASE_URL


def get_embeddings_model(
    model: str = None,
) -> Optional[Embeddings]:
    """Get embeddings model.

    Supports:
    - ollama/nomic-embed-text: Local Ollama embeddings with 2K context window (default)
    - openai/*: OpenAI embeddings
    - weaviate/*: Legacy Weaviate built-in vectorizer (deprecated)
    """
    if model is None:
        model = os.getenv("EMBEDDING_MODEL", "ollama/nomic-embed-text")

    provider, model_name = model.split("/", maxsplit=1)

    if provider == "ollama":
        # Ollama embeddings with nomic-embed-text (2K context, 768 dimensions)
        return OllamaEmbeddings(
            model=model_name,
            base_url=OLLAMA_BASE_URL,
        )
    elif provider == "openai":
        return OpenAIEmbeddings(model=model_name, chunk_size=200)
    elif provider == "weaviate":
        # Weaviate's built-in vectorizer handles embeddings internally
        return None
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
