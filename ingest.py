"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from bs4 import BeautifulSoup, SoupStrainer, Tag
import weaviate
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain.vectorstores import Weaviate

from constants import (
    WEAVIATE_DOCS_INDEX_NAME,
)

logger = logging.getLogger(__name__)

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]


def _get_text(tag):
    new_line_elements = {"h1", "h2", "h3", "h4", "code", "p", "li"}
    code_elements = {"code"}
    skip_elements = {"button"}
    for child in tag.children:
        if isinstance(child, Tag):
            # if the tag is a block type tag then yield new lines before after
            is_code_element = child.name in code_elements
            is_block_element = is_code_element and "codeBlockLines_e6Vv" in child.get(
                "class", ""
            )
            if is_block_element:
                yield "\n```python\n"
            elif is_code_element:
                yield "`"
            elif child.name in new_line_elements:
                yield "\n"
            if child.name == "br":
                yield from ["\n"]
            elif child.name not in skip_elements:
                yield from _get_text(child)

            if is_block_element:
                yield "```\n"
            elif is_code_element:
                yield "`"
        else:
            yield child.text


def _doc_extractor(html):
    soup = BeautifulSoup(html, "lxml", parse_only=SoupStrainer("article"))
    for tag in soup.find_all(["nav", "footer", "aside"]):
        tag.decompose()
    joined = "".join(_get_text(soup))
    return re.sub(r"\n\n+", "\n\n", joined)


def _simple_extractor(html):
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text)


def ingest_docs():
    simple_urls = ["https://api.python.langchain.com/en/latest/"]
    doc_urls = [
        "https://python.langchain.com/docs/get_started",
        "https://python.langchain.com/docs/use_cases",
        "https://python.langchain.com/docs/integrations",
        "https://python.langchain.com/docs/modules",
        "https://python.langchain.com/docs/guides",
        "https://python.langchain.com/docs/additional_resources",
        "https://python.langchain.com/docs/community",
        "https://python.langchain.com/docs/expression_language",
    ]
    urls = [(url, _simple_extractor) for url in simple_urls] + [
        (url, _doc_extractor) for url in doc_urls
    ]
    # Drop trailing "/" to avoid duplicate documents.
    link_regex = (
        f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
        f"(?:[\#'\"]|\/[\#'\"])"
    )
    documents = []
    for url, extractor in urls:
        documents += RecursiveUrlLoader(
            url=url,
            max_depth=8,
            extractor=extractor,
            prevent_outside=True,
            use_async=True,
            timeout=600,
            link_regex=link_regex,
            check_response_status=True,
        ).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(documents)
    # We try to return 'source' and 'title' metadata when querying vector store and
    # Weaviate will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

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
        attributes=["source", "title"],
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
