import os
from contextlib import contextmanager
from typing import Iterator

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_weaviate import WeaviateVectorStore

from backend.configuration import BaseConfiguration
from backend.constants import (
    OLLAMA_BASE_URL,
    WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
)
from backend.utils import get_weaviate_client

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "https://weaviate.hanu-nus.com")
WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY", "admin-key")


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model_name = model.split("/", maxsplit=1)
    match provider:
        case "ollama":
            from langchain_ollama import OllamaEmbeddings

            return OllamaEmbeddings(
                model=model_name,
                base_url=OLLAMA_BASE_URL,
            )
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model_name)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


@contextmanager
def make_weaviate_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Iterator[BaseRetriever]:
    with get_weaviate_client(
        weaviate_url=WEAVIATE_URL,
        weaviate_api_key=WEAVIATE_API_KEY,
        grpc_port=50051,
    ) as weaviate_client:
        store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
            text_key="text",
            embedding=embedding_model,
            attributes=["source", "title"],
        )
        search_kwargs = {**configuration.search_kwargs, "return_uuids": True}
        yield store.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Iterator[BaseRetriever]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "weaviate":
            with make_weaviate_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
