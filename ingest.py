"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
from bs4 import BeautifulSoup
import weaviate
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_transformers import Html2TextTransformer

from constants import (
    WEAVIATE_DOCS_INDEX_NAME,
)

logger = logging.getLogger(__name__)

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]


def ingest_docs():
    urls = [
        "https://api.python.langchain.com/en/latest/api_reference.html#module-langchain",
        "https://python.langchain.com/docs/get_started",
        "https://python.langchain.com/docs/use_cases",
        "https://python.langchain.com/docs/integrations",
        "https://python.langchain.com/docs/modules",
        "https://python.langchain.com/docs/guides",
        "https://python.langchain.com/docs/ecosystem",
        "https://python.langchain.com/docs/additional_resources",
        "https://python.langchain.com/docs/community",
    ]

    documents = []
    for url in urls:
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=8,
            extractor=lambda x: BeautifulSoup(x, "lxml").text,
            prevent_outside=True,
        )
        temp_docs = loader.load()
        temp_docs = [doc for i, doc in enumerate(temp_docs) if doc not in temp_docs[:i]]
        documents += temp_docs

    html2text = Html2TextTransformer()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    docs_transformed = html2text.transform_documents(documents)
    docs_transformed = text_splitter.split_documents(docs_transformed)

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    embedding = OpenAIEmbeddings(chunk_size=200)  # rate limit
    vectorstore = Weaviate(
        client,
        WEAVIATE_DOCS_INDEX_NAME,
        "text",
        embedding=embedding,
        by_text=False,
        attributes=["source"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()
    index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source",
    )

    logger.info(
        "LangChain now has this many vectors: ",
        client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do(),
    )


if __name__ == "__main__":
    ingest_docs()
