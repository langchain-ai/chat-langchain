"""Clear Weaviate index."""

import logging
import os

import weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.vectorstores import Weaviate

logger = logging.getLogger(__name__)

WEAVIATE_URL = os.environ.get("WEAVIATE_URL", "http://localhost:8080")
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]
WEAVIATE_DOCS_INDEX_NAME = "LangChain_Combined_Docs_transformers_multi_qa_MiniLM"


def clear():
    # Note: This uses the old Weaviate client API v3. Consider upgrading to v4.
    host = WEAVIATE_URL.replace("http://", "").replace("https://", "")
    client = weaviate.Client(
        url=WEAVIATE_URL,
    )
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=OpenAIEmbeddings(),
        by_text=False,
        attributes=["source", "title"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    indexing_stats = index(
        [],
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
    )

    logger.info("Indexing stats: ", indexing_stats)
    logger.info(
        "LangChain now has this many vectors: ",
        client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do(),
    )


if __name__ == "__main__":
    clear()
