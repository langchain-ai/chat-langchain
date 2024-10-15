import contextlib
import os
from typing import Iterator

import weaviate
from langchain_core.retrievers import BaseRetriever
from langchain_weaviate import WeaviateVectorStore

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.embeddings import get_embeddings_model


@contextlib.contextmanager
def get_retriever() -> Iterator[BaseRetriever]:
    # TODO: we probably want to mimic the logic from the template more closely here
    # to make it easier to experiment with different retrievers
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
            embedding=get_embeddings_model(),
            attributes=["source", "title"],
        )
        k = k or 6
        yield store.as_retriever(search_kwargs=dict(k=k))