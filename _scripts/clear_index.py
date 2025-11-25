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
    OLLAMA_BASE_EMBEDDING_DOCS_URL,
    OLLAMA_BASE_URL,
    WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
)
from backend.utils import get_weaviate_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECORD_MANAGER_DB_URL = os.environ.get(
    "RECORD_MANAGER_DB_URL",
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
        collection_name = WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME

        # First, directly delete all documents from Weaviate collection
        # This ensures we delete everything, not just what's tracked in record manager
        try:
            collection = weaviate_client.collections.get(collection_name)

            # Get count before deletion
            initial_count = collection.aggregate.over_all().total_count
            logger.info(
                f"Found {initial_count} documents in collection before deletion"
            )

            if initial_count > 0:
                # Fetch all object UUIDs and delete them individually
                # This is the most reliable way to delete all documents
                import weaviate.classes.query as wq

                deleted_count = 0
                batch_size = 100

                while True:
                    # Fetch a batch of objects (only get UUIDs, not full data)
                    objects = collection.query.fetch_objects(limit=batch_size)

                    if not objects.objects:
                        break

                    # Delete each object individually
                    for obj in objects.objects:
                        try:
                            collection.data.delete_by_id(obj.uuid)
                            deleted_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to delete object {obj.uuid}: {e}")

                    logger.info(
                        f"Deleted batch of {len(objects.objects)} documents (total: {deleted_count})"
                    )

                    # If we got fewer objects than batch_size, we're done
                    if len(objects.objects) < batch_size:
                        break

                logger.info(
                    f"Successfully deleted {deleted_count} documents directly from Weaviate collection: {collection_name}"
                )
            else:
                logger.info("Collection is already empty")

        except Exception as e:
            logger.warning(f"Could not delete directly from collection: {e}")
            logger.info("Falling back to record manager cleanup...")

        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=collection_name,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title"],
        )

        record_manager = SQLRecordManager(
            f"weaviate/{collection_name}",
            db_url=RECORD_MANAGER_DB_URL,
        )

        record_manager.create_schema()

        # Also clean up record manager to keep it in sync
        indexing_stats = index(
            [],
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",
        )

        logger.info(f"Indexing stats: {indexing_stats}")
        num_vecs = (
            weaviate_client.collections.get(collection_name)
            .aggregate.over_all()
            .total_count
        )
        logger.info(
            f"General Guides and Tutorials now has this many vectors: {num_vecs}"
        )


if __name__ == "__main__":
    clear()
