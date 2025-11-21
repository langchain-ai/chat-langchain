from contextlib import contextmanager
from typing import Iterator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import os
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_weaviate import WeaviateVectorStore

from backend.configuration import BaseConfiguration
from backend.constants import (
    OLLAMA_BASE_URL,
    WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
)
from backend.embeddings import get_embeddings_model
from backend.utils import get_weaviate_client

RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_GRPC_URL = os.environ.get("WEAVIATE_GRPC_URL")

WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")


@contextmanager
def make_weaviate_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Iterator[BaseRetriever]:
    with get_weaviate_client(
        weaviate_url=WEAVIATE_URL,
        weaviate_grpc_url=WEAVIATE_GRPC_URL,
        weaviate_api_key=WEAVIATE_API_KEY,
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
    base_url: str = OLLAMA_BASE_URL,
) -> Iterator[BaseRetriever]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = get_embeddings_model(
        configuration.embedding_model, base_url=base_url
    )
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
