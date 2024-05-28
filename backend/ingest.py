"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os

import weaviate
from bs4 import BeautifulSoup, Comment
from constants import WEAVIATE_DOCS_INDEX_NAME
from langchain.document_loaders import RecursiveUrlLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_community.vectorstores import Weaviate
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


# def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
#     title = soup.find("title")
#     description = soup.find("meta", attrs={"name": "description"})
#     html = soup.find("html")
#     return {
#         "source": meta["loc"],
#         "title": title.get_text() if title else "",
#         "description": description.get("content", "") if description else "",
#         "language": html.get("lang", "") if html else "",
#         **meta,
#     }


# def load_langchain_docs():
#     return SitemapLoader(
#         "https://python.langchain.com/sitemap.xml",
#         filter_urls=["https://python.langchain.com/"],
#         parsing_function=langchain_docs_extractor,
#         default_parser="lxml",
#         bs_kwargs={
#             "parse_only": SoupStrainer(
#                 name=("article", "title", "html", "lang", "content")
#             ),
#         },
#         meta_function=metadata_extractor,
#     ).load()


def load_looker_docs():
    return RecursiveUrlLoader(
        url="https://cloud.google.com/looker/docs/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?![^\"]*\?hl=)(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    """
    Extracts and formats text from HTML content.

    This function processes the given HTML content to remove unwanted tags (script, style, meta, and link),
    HTML comments, and formats certain elements like <strong> tags. It also handles nested elements and
    formats tables using the  function. Hyperlinks are simplified to display just the text.

    Parameters:
    - html (str): A string containing the HTML content to be processed.

    Returns:
    - str: The extracted and formatted text from the HTML content. If the main content div is not found,
           returns a message indicating that the main content was not found.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Find the main content div
    main_content = soup.find("devsite-content")

    if main_content:
        # Remove unwanted tags
        for script_or_style in main_content(["script", "style", "meta", "link"]):
            script_or_style.decompose()

        # Remove HTML comments
        for comment in main_content.find_all(
            string=lambda text: isinstance(text, Comment)
        ):
            comment.extract()

        # Replace <strong> tags with asterisks for bold formatting
        for strong_tag in main_content.find_all("strong"):
            strong_tag.replace_with(f"**{strong_tag.text}**")

        extracted_text = []

        # Recursive function to handle nested elements
        def process_element(element):
            if element.name == "table":
                extracted_text.append(format_table(element))
            else:
                for child in element.children:
                    if child.name:
                        if child.name == "a":
                            # Handle hyperlinks
                            link_text = child.get_text(strip=True)
                            extracted_text.append(f"{link_text}")
                        elif child.name in ["p", "br"]:
                            extracted_text.append("\n")
                            process_element(child)
                        elif child.name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                            extracted_text.append(f"\n\n{child.get_text(strip=True)}")
                        elif child.name in ["button"]:
                            pass
                        elif child.name == "devsite-feature-tooltip":
                            pass
                        elif child.name == "devsite-content-footer":
                            pass
                        elif (
                            child.name == "div"
                            and "class" in child.attrs
                            and "devsite-content-data" in child["class"]
                        ):
                            pass
                        else:
                            process_element(child)
                    elif child.string:
                        text = child.string.strip()
                        if text:
                            extracted_text.append(text + " ")

        # Process the main content recursively
        process_element(main_content)

        # Combine extracted text while preserving necessary spaces and new lines
        combined_text = "".join(extracted_text).replace("\n ", "\n").strip()

        return combined_text
    else:
        return "Main content not found."


def format_table(table) -> str:
    rows = table.find_all("tr")
    if not rows:
        return ""

    # Extract headers if present
    headers = rows[0].find_all("th")
    if headers:
        header_text = (
            "| " + " | ".join(cell.get_text(strip=True) for cell in headers) + " |"
        )
        separator = "| " + " | ".join("---" for _ in headers) + " |"
        body_start_idx = 1
    else:
        header_text = ""
        separator = ""
        body_start_idx = 0

    # Extract body rows
    body_rows = rows[body_start_idx:]
    body_text = ""
    for row in body_rows:
        cells = row.find_all(["td", "th"])
        row_text = "| " + " | ".join(cell.get_text(strip=True) for cell in cells) + " |"
        body_text += row_text + "\n"

    # Combine header and body
    if headers:
        table_text = header_text + "\n" + separator + "\n" + body_text
    else:
        table_text = body_text

    return "\n" + table_text.strip() + "\n"


# def load_api_docs():
#     return RecursiveUrlLoader(
#         url="https://api.python.langchain.com/en/latest/",
#         max_depth=8,
#         extractor=simple_extractor,
#         prevent_outside=True,
#         use_async=True,
#         timeout=600,
#         # Drop trailing / to avoid duplicate pages.
#         link_regex=(
#             f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
#             r"(?:[\#'\"]|\/[\#'\"])"
#         ),
#         check_response_status=True,
#         exclude_dirs=(
#             "https://api.python.langchain.com/en/latest/_sources",
#             "https://api.python.langchain.com/en/latest/_modules",
#         ),
#     ).load()


def ingest_docs():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    vectorstore = Weaviate(
        client=client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=embedding,
        by_text=False,
        attributes=["source", "title"],
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    # docs_from_documentation = load_langchain_docs()
    # logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    # docs_from_api = load_api_docs()
    # logger.info(f"Loaded {len(docs_from_api)} docs from API")
    docs_from_looker = load_looker_docs()
    logger.info(f"Loaded {len(docs_from_looker)} docs from Langsmith")

    docs_transformed = text_splitter.split_documents(docs_from_looker)
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
        cleanup="incremental",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")

    num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    logger.info(
        f"LangChain now has this many vectors: {num_vecs}",
    )


if __name__ == "__main__":
    ingest_docs()
