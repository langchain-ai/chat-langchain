"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from typing import Optional

import weaviate
from weaviate.classes.init import Auth
from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain_community.document_loaders import JSONLoader
from langchain.indexes import SQLRecordManager, index
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
#from langchain_community.vectorstores import Chroma

from constants import WEAVIATE_DOCS_INDEX_NAME
from embeddings import get_embeddings_model
from parser import langchain_docs_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def metadata_extractor(
    meta: dict, soup: BeautifulSoup, title_suffix: Optional[str] = None
) -> dict:
    title_element = soup.find("title")
    description_element = soup.find("meta", attrs={"name": "description"})
    html_element = soup.find("html")
    title = title_element.get_text() if title_element else ""
    if title_suffix is not None:
        title += title_suffix

    return {
        "source": meta["loc"],
        "title": title,
        "description": description_element.get("content", "")
        if description_element
        else "",
        "language": html_element.get("lang", "") if html_element else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langgraph_docs():
    return SitemapLoader(
        "https://langchain-ai.github.io/langgraph/sitemap.xml",
        parsing_function=simple_extractor,
        default_parser="lxml",
        bs_kwargs={"parse_only": SoupStrainer(name=("article", "title"))},
        meta_function=lambda meta, soup: metadata_extractor(
            meta, soup, title_suffix=" | 🦜🕸️LangGraph"
        ),
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str | BeautifulSoup) -> str:
    if isinstance(html, str):
        soup = BeautifulSoup(html, "lxml")
    elif isinstance(html, BeautifulSoup):
        soup = html
    else:
        raise ValueError(
            "Input should be either BeautifulSoup object or an HTML string"
        )
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


# function for custom metadata
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["title"] = record.get("article_title")
    metadata["publish_date"] = record.get("publish_date")
    metadata["claim_author"] = record.get("claim_author")
    metadata["rating"] = record.get("rating")
    metadata["source"] = record.get("url")

    return metadata


def load_vf_docs():
    SEEK_DATASET = os.environ["SEEK_DATASET"]
    return JSONLoader(
        file_path=SEEK_DATASET,
        jq_schema='.[].data[]',
        content_key="post_content",
        metadata_func=metadata_func
    ).load()


def ingest_docs():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    # RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    DATABASE_HOST = os.environ["DATABASE_HOST"]
    DATABASE_PORT = os.environ["DATABASE_PORT"]
    DATABASE_USERNAME = os.environ["DATABASE_USERNAME"]
    DATABASE_PASSWORD = os.environ["DATABASE_PASSWORD"]
    DATABASE_NAME = os.environ["DATABASE_NAME"]
    RECORD_MANAGER_DB_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    ) as weaviate_client:
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=embedding,
            attributes=["source", "title"],
        )

        COLLECTION_NAME = os.environ["COLLECTION_NAME"]
        
        #chroma_client = chromadb.HttpClient(
        #    host=DATABASE_HOST,
        #    port="9010"
        #)
        #vectorstore = Chroma(
        #    client=chroma_client,
        #    collection_name=COLLECTION_NAME,
        #    embedding_function=embedding,
            #persist_directory="./chroma_chat_langchain_test_db",
        #)
        #vectorstore.persist()

        # record_manager = SQLRecordManager(
        #     f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        # )

        record_manager = SQLRecordManager(
            f"weaviate/{COLLECTION_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
        record_manager.create_schema()

            #docs_from_documentation = load_langchain_docs()
            #logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
            #docs_from_api = load_api_docs()
            #logger.info(f"Loaded {len(docs_from_api)} docs from API")
            #docs_from_langsmith = load_langsmith_docs()
            #logger.info(f"Loaded {len(docs_from_langsmith)} docs from LangSmith")
            #docs_from_langgraph = load_langgraph_docs()
            #logger.info(f"Loaded {len(docs_from_langgraph)} docs from LangGraph")
        docs_from_vf = load_vf_docs()
        logger.info(f"Loaded {len(docs_from_vf)} docs from VF")

        docs_transformed = text_splitter.split_documents(
            docs_from_vf

        )
        docs_transformed = [
            doc for doc in docs_transformed if len(doc.page_content) > 10
        ]

            # We try to return 'source' and 'title' metadata when querying vector store and
            # Weaviate will error at query time if one of the attributes is missing from a
            # retrieved document.
        for doc in docs_transformed:
            if "source" not in doc.metadata:
                doc.metadata["source"] = ""
            if "title" not in doc.metadata:
                doc.metadata["title"] = ""

        indexing_stats = index(
            docs_transformed,
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",
            force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
        )

        logger.info(f"Indexing stats: {indexing_stats}")
        num_vecs = (
            weaviate_client.collections.get(WEAVIATE_DOCS_INDEX_NAME)
            .aggregate.over_all()
            .total_count
        )
        #logger.info(
        #    f"NUMBER OF DOCUMENTS: {len(vectorstore.get()['documents'])}"
        #)


if __name__ == "__main__":
    ingest_docs()
