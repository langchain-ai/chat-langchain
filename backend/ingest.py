"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from typing import Optional

from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from pymilvus import connections, utility

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
# from backend.constants import WEAVIATE_DOCS_INDEX_NAME # No longer used
from backend.parser import langchain_docs_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    if embedding_provider == "ollama":
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
        logger.info(
            f"Using Ollama Embeddings with base_url='{ollama_base_url}' and model='{ollama_embedding_model}'"
        )
        return OllamaEmbeddings(
            base_url=ollama_base_url, model=ollama_embedding_model
        )
    elif embedding_provider == "openai":
        logger.info("Using OpenAI Embeddings model='text-embedding-3-small'")
        return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)
    else:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER: {embedding_provider}. Must be 'openai' or 'ollama'."
        )


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
        "https://python.langchain.com/v0.2/sitemap.xml",
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


def load_knowledge_base_pdfs(docs_path_str: str = "knowledge_docs") -> list:
    """
    Loads PDF documents from a specified directory.

    Args:
        docs_path_str: The path to the folder containing PDF files.

    Returns:
        A list of Langchain Document objects, where each Document represents a page.
    """
    docs_path = Path(docs_path_str)
    if not docs_path.is_dir():
        logger.warning(
            f"Knowledge base PDF directory '{docs_path}' not found. Skipping PDF loading."
        )
        return []

    loaded_documents = []
    for pdf_file_path in docs_path.glob("*.pdf"):
        logger.info(f"Processing PDF file: {pdf_file_path.name}")
        try:
            loader = PyPDFLoader(str(pdf_file_path))
            pages = loader.load()

            # TODO: User needs to implement logic to extract the source URL from the PDF content/structure.
            # This could involve parsing the PDF for a URL, using a sidecar file, or a naming convention.
            extracted_url = "placeholder_url_needs_implementation"
            logger.warning(
                f"Using placeholder URL ('{extracted_url}') for {pdf_file_path.name}. "
                "Implement robust URL extraction logic for production use."
            )

            for page in pages:
                page.metadata['source'] = extracted_url
                page.metadata['title'] = pdf_file_path.stem # Use stem for filename without extension
                loaded_documents.append(page)
            logger.info(f"Loaded {len(pages)} pages from {pdf_file_path.name}")
        except Exception as e:
            logger.error(f"Error loading or processing PDF file {pdf_file_path.name}: {e}")
            # Optionally, re-raise or handle more gracefully
            # raise e 

    return loaded_documents


def ingest_docs():
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "knowledge_base")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    logger.info(f"Attempting to connect to Milvus: host={MILVUS_HOST}, port={MILVUS_PORT}")
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info(f"Successfully connected to Milvus.")

    vectorstore = Milvus(
        embedding_function=embedding,
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
        auto_id=True,
        primary_field="id", 
        text_field="text", 
        vector_field="embedding",
    )

    # Construct the record manager namespace using the Milvus collection name
    # to keep it distinct from potential Weaviate namespaces if DB is reused.
    record_manager_namespace = f"milvus/{MILVUS_COLLECTION_NAME}"
    record_manager = SQLRecordManager(
        record_manager_namespace, db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    knowledge_base_docs = load_knowledge_base_pdfs()
    logger.info(f"Loaded {len(knowledge_base_docs)} pages from PDF documents.")

    # Comment out existing data loading (Web Loaders)
    # docs_from_documentation = load_langchain_docs()
    # logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    # docs_from_api = load_api_docs()
    # logger.info(f"Loaded {len(docs_from_api)} docs from API")
    # docs_from_langsmith = load_langsmith_docs()
    # logger.info(f"Loaded {len(docs_from_langsmith)} docs from LangSmith")
    # docs_from_langgraph = load_langgraph_docs()
    # logger.info(f"Loaded {len(docs_from_langgraph)} docs from LangGraph")

    # docs_transformed = text_splitter.split_documents(
    #     docs_from_documentation
    #     + docs_from_api
    #     + docs_from_langsmith
    #     + docs_from_langgraph
    # )
    # docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # Initialize all_docs with PDF documents.
    # If other loaders (e.g., web loaders) are re-enabled, concatenate their results here.
    all_docs = knowledge_base_docs
    # For example, if web loaders were active:
    # all_docs = knowledge_base_docs + docs_from_documentation + docs_from_api + ...

    logger.info(f"Total documents loaded for processing: {len(all_docs)}")
    
    docs_transformed = text_splitter.split_documents(all_docs)
    docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]

    # We try to return 'source' and 'title' metadata when querying vector store.
    # Ensure these fields are present in document metadata.
    # Milvus client for Langchain should handle mapping these to its schema if possible.
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
    
    # Check Milvus collection stats
    if utility.has_collection(MILVUS_COLLECTION_NAME):
        stats = utility.get_collection_stats(MILVUS_COLLECTION_NAME)
        logger.info(f"Milvus collection '{MILVUS_COLLECTION_NAME}' has {stats['row_count']} vectors.")
    else:
        logger.info(f"Milvus collection '{MILVUS_COLLECTION_NAME}' does not exist or is empty.")
    
    # Remove the constant WEAVIATE_DOCS_INDEX_NAME as it's no longer used
    # This should be done carefully if it's used elsewhere, but based on the
    # current context, it seems specific to Weaviate.
    # For now, we will assume it's not used elsewhere and can be removed from constants.py later.
    # Also, the import of WEAVIATE_DOCS_INDEX_NAME from backend.constants can be removed.
    # The line referring to WEAVIATE_DOCS_INDEX_NAME in imports has been commented out.


if __name__ == "__main__":
    ingest_docs()
