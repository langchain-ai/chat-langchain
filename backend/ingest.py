"""Load html from files, clean up, split, ingest into Weaviate."""

import json
import logging
import os
import re
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore

from backend.constants import (
    OLLAMA_BASE_EMBEDDING_DOCS_URL,
    OLLAMA_BASE_URL,
    WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME,
)
from backend.embeddings import get_embeddings_model
from backend.parser import langchain_docs_extractor
from backend.utils import get_weaviate_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
WEAVIATE_GRPC_URL = os.environ.get("WEAVIATE_GRPC_URL")

WEAVIATE_API_KEY = os.environ.get("WEAVIATE_API_KEY")


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
        "description": (
            description_element.get("content", "") if description_element else ""
        ),
        "language": html_element.get("lang", "") if html_element else "",
        **meta,
    }


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


#########################
# General Guides and Tutorials
#########################


# NOTE: To be deprecated once LangChain docs are migrated to new site.
def load_langchain_python_docs():
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


# NOTE: To be deprecated once LangChain docs are migrated to new site.
def load_langchain_js_docs():
    return SitemapLoader(
        "https://js.langchain.com/sitemap.xml",
        parsing_function=simple_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            )
        },
        meta_function=metadata_extractor,
        filter_urls=["https://js.langchain.com/docs/concepts"],
    ).load()


def load_aggregated_docs_site():
    return SitemapLoader(
        "https://docs.langchain.com/sitemap.xml",
        parsing_function=simple_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            )
        },
        meta_function=metadata_extractor,
        filter_urls=[
            "https://docs.langchain.com/oss/javascript/langchain/mcp",
            "https://docs.langchain.com/oss/javascript/langchain/agents",
            "https://docs.langchain.com/oss/javascript/langchain/context-engineering",
        ],
    ).load()


def ingest_general_guides_and_tutorials():  # test with just js docs
    # langchain_python_docs = load_langchain_python_docs()
    # langchain_js_docs = load_langchain_js_docs()
    aggregated_site_docs = load_aggregated_docs_site()
    return aggregated_site_docs  # + langchain_js_docs


def ingest_docs():
    # Larger chunks for nomic-embed-text (2K token context window)
    # 4000 chars ≈ 1000-1300 tokens, well within the 2K limit
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model(base_url=OLLAMA_BASE_EMBEDDING_DOCS_URL)

    with get_weaviate_client(
        weaviate_url=WEAVIATE_URL,
        weaviate_grpc_url=WEAVIATE_GRPC_URL,
        weaviate_api_key=WEAVIATE_API_KEY,
    ) as weaviate_client:
        # General Guides and Tutorials
        general_guides_and_tutorials_vectorstore = WeaviateVectorStore(
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
        general_guides_and_tutorials_docs = ingest_general_guides_and_tutorials()

        print(f"General Guides and Tutorials docs: {general_guides_and_tutorials_docs}")

        # Write raw documents to JSON file for inspection
        raw_docs_data = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "type": doc.type if hasattr(doc, "type") else "Document",
            }
            for doc in general_guides_and_tutorials_docs
        ]

        raw_docs_file_path = os.path.join(
            os.path.dirname(__file__), "..", "raw_docs.json"
        )
        with open(raw_docs_file_path, "w", encoding="utf-8") as f:
            json.dump(raw_docs_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote {len(raw_docs_data)} raw documents to raw_docs.json")

        docs_transformed = text_splitter.split_documents(
            general_guides_and_tutorials_docs
        )
        docs_transformed = [
            doc for doc in docs_transformed if len(doc.page_content) > 10
        ]

        # Write transformed chunks to JSON file for inspection
        chunks_data = []
        for doc in docs_transformed:
            doc_dict = {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
                "type": doc.type if hasattr(doc, "type") else "Document",
            }
            # Add any additional fields that might exist
            for attr in dir(doc):
                if not attr.startswith("_") and attr not in [
                    "page_content",
                    "metadata",
                    "type",
                ]:
                    try:
                        value = getattr(doc, attr)
                        if not callable(value):
                            # Try to serialize the value, convert to string if not serializable
                            try:
                                json.dumps(value)
                                doc_dict[attr] = value
                            except (TypeError, ValueError):
                                # Skip non-serializable objects like FieldInfo
                                pass
                    except Exception:
                        pass
            chunks_data.append(doc_dict)

        chunks_file_path = os.path.join(os.path.dirname(__file__), "..", "chunks.json")
        with open(chunks_file_path, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote {len(chunks_data)} chunks to chunks.json")

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
            general_guides_and_tutorials_vectorstore,
            cleanup="full",
            source_id_key="source",
            force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
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
            f"General Guides and Tutorials now has this many vectors: {num_vecs}",
        )


if __name__ == "__main__":
    ingest_docs()
