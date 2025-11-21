"""Clear Weaviate index."""

import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langchain.indexes import SQLRecordManager, index
from langchain_weaviate import WeaviateVectorStore
from backend.embeddings import get_embeddings_model
from backend.constants import (
    OLLAMA_BASE_URL,
    WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
)
from backend.utils import get_weaviate_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECORD_MANAGER_DB_URL = os.environ.get(
    "RECORD_MANAGER_DB_URL",
    "postgresql://postgres:zkdtn1234@localhost:5432/chat_langchain",
)

RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_GRPC_URL = os.environ.get("WEAVIATE_GRPC_URL")

WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")


def clear():
    embedding = get_embeddings_model(base_url=OLLAMA_BASE_URL)

    with get_weaviate_client(
        weaviate_url=WEAVIATE_URL,
        weaviate_grpc_url=WEAVIATE_GRPC_URL,
        weaviate_api_key=WEAVIATE_API_KEY,
    ) as weaviate_client:
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title"],
        )

        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME}",
            db_url=RECORD_MANAGER_DB_URL,
        )

        record_manager.create_schema()

        indexing_stats = index(
            [],
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",
        )

        logger.info(f"Indexing stats: {indexing_stats}")
        num_vecs = (
            weaviate_client.collections.get(
                WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME
            )
            .aggregate.over_all()
            .total_count
        )
        logger.info(
            f"General Guides and Tutorials now has this many vectors: {num_vecs}"
        )


if __name__ == "__main__":
    clear()
