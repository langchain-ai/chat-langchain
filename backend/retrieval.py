import os
from contextlib import contextmanager
from typing import Any, Iterator, List, Optional

import weaviate
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_weaviate import WeaviateVectorStore

from backend.configuration import BaseConfiguration
from backend.constants import WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME


class WeaviateNativeRetriever(BaseRetriever):
    """Retriever that uses Weaviate's built-in vectorizer with near_text queries.

    This retriever is used when embedding=None, allowing Weaviate to handle
    text vectorization server-side using its built-in text2vec-transformers module.
    """

    client: Any
    index_name: str
    search_kwargs: dict = {}

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query using Weaviate's near_text query."""
        k = self.search_kwargs.get("k", 6)

        collection = self.client.collections.get(self.index_name)

        # Use near_text - Weaviate vectorizes the query using built-in model
        response = collection.query.near_text(
            query=query, limit=k, return_metadata=["distance"]
        )

        # Convert Weaviate objects to LangChain Documents
        docs = []
        for obj in response.objects:
            doc = Document(
                page_content=obj.properties.get("text", ""),
                metadata={
                    "source": obj.properties.get("source", ""),
                    "title": obj.properties.get("title", ""),
                    "distance": obj.metadata.distance,
                    "uuid": str(obj.uuid),
                },
            )
            docs.append(doc)

        return docs


def make_text_encoder(model: str) -> Optional[Embeddings]:
    """Connect to the configured text encoder.

    Returns None for Weaviate's built-in vectorizer.
    """
    provider, model_name = model.split("/", maxsplit=1)
    match provider:
        case "weaviate":
            # Weaviate's built-in vectorizer handles embeddings internally
            return None
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model_name)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


@contextmanager
def make_weaviate_retriever(
    configuration: BaseConfiguration, embedding_model: Optional[Embeddings]
) -> Iterator[BaseRetriever]:
    weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
    host = weaviate_url.replace("http://", "").replace("https://", "")

    with weaviate.connect_to_local() as weaviate_client:
        # If embedding_model is None, use Weaviate's native near_text query
        # which allows server-side vectorization with built-in text2vec-transformers
        if embedding_model is None:
            search_kwargs = {
                **configuration.search_kwargs,
                "k": configuration.search_kwargs.get("k", configuration.k),
            }
            retriever = WeaviateNativeRetriever(
                client=weaviate_client,
                index_name=WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
                search_kwargs=search_kwargs,
            )
            yield retriever
        else:
            # Use LangChain's WeaviateVectorStore for external embeddings
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
