import os
from contextlib import contextmanager
from typing import Iterator

import weaviate
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_weaviate import WeaviateVectorStore
# from langchain_community.vectorstores import Chroma

from backend.configuration import BaseConfiguration
from backend.constants import WEAVIATE_DOCS_INDEX_NAME

DATABASE_HOST = os.environ["DATABASE_HOST"]
# COLLECTION_NAME = os.environ["COLLECTION_NAME"]


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case "voyageai":
            from langchain_voyageai import VoyageAIEmbeddings

            return VoyageAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


@contextmanager
def make_weaviate_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Iterator[BaseRetriever]:
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.classes.init.Auth.api_key(
            os.environ.get("WEAVIATE_API_KEY", "not_provided")
        ),
        skip_init_checks=True,
    ) as weaviate_client:
        store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=embedding_model,
            attributes=["source", "title"],
        )
        search_kwargs = {**configuration.search_kwargs, "return_uuids": True}
        yield store.as_retriever(
            # search_type="similarity_score_threshold",
            # search_kwargs={'k': 20, 'score_threshold': 0.60, 'return_uuids': True},
            search_kwargs=search_kwargs
        )

    # chroma_client = chromadb.HttpClient(
    #    host=DATABASE_HOST,
    #    port="9010"
    # )
    # vectorstore = Chroma(
    #    client=chroma_client,
    #    collection_name=COLLECTION_NAME,
    #    embedding_function=embedding_model,
    #    #persist_directory="./chroma_chat_langchain_test_db",
    # )
    # yield vectorstore.as_retriever(search_kwargs=dict(k=4))


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
