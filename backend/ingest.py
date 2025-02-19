"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from my_parser import langchain_docs_extractor

import weaviate
from weaviate.classes.init import Auth
from bs4 import BeautifulSoup, SoupStrainer
from my_constants import WEAVIATE_DOCS_INDEX_NAME
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader, DirectoryLoader, TextLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_weaviate import WeaviateVectorStore
from langchain_core.embeddings import Embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

def get_embeddings_model() -> Embeddings:
    return OllamaEmbeddings(model="nomic-embed-text")


def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }

def load_directory_docs():
    return DirectoryLoader("data/", glob="**/*.rst", loader_cls=TextLoader, use_multithreading=True, recursive=True).load()

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


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
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


def ingest_docs():
    # DATABASE_HOST = os.environ["DATABASE_HOST"]
    # DATABASE_PORT = os.environ["DATABASE_PORT"]
    # DATABASE_USERNAME = os.environ["DATABASE_USERNAME"]
    # DATABASE_PASSWORD = os.environ["DATABASE_PASSWORD"]
    # DATABASE_NAME = os.environ["DATABASE_NAME"]
    # RECORD_MANAGER_DB_URL = f"postgresql://{DATABASE_USERNAME}:{DATABASE_PASSWORD}@{DATABASE_HOST}:{DATABASE_PORT}/{DATABASE_NAME}"

    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    # client = weaviate.Client(
    #     url=WEAVIATE_URL,
    #     auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    # )
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    ) as weaviate_client:
        print(weaviate_client.is_ready())
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
                embedding=embedding,
                attributes=["source", "title"],
            )

        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
        record_manager.create_schema()
        # docs_from_documentation = load_langchain_docs()
        # logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")\
        docs_from_directory  = load_directory_docs()
        logger.info(f"Loaded {len(docs_from_directory)} docs from directory")\
        # docs_from_api = load_api_docs()
        # logger.info(f"Loaded {len(docs_from_api)} docs from API")
        # docs_from_langsmith = load_langsmith_docs()
        # logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")

        docs_transformed = text_splitter.split_documents(
            docs_from_directory 
            # docs_from_documentation + docs_from_api + docs_from_langsmith
        )
        docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

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
        print("shoval is king")
        logger.info(f"Indexing stats: {indexing_stats}")
        num_vecs = (
            weaviate_client.collections.get(WEAVIATE_DOCS_INDEX_NAME)
            .aggregate.over_all()
            .total_count
        )
        logger.info(
            f"LangChain now has this many vectors: {num_vecs}",
        )


if __name__ == "__main__":
    ingest_docs()
