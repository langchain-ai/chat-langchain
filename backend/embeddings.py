from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def get_embeddings_model(
    model: str = "weaviate/text2vec-transformers",
) -> Optional[Embeddings]:
    """Get embeddings model.

    Returns None for Weaviate's built-in vectorizer (text2vec-transformers),
    as Weaviate handles embeddings internally.
    """
    provider, model_name = model.split("/", maxsplit=1)

    if provider == "weaviate":
        # Weaviate's built-in vectorizer handles embeddings internally
        return None
    elif provider == "openai":
        return OpenAIEmbeddings(model=model_name, chunk_size=200)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")
