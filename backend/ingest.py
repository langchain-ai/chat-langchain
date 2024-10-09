"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
import requests
from typing import Optional

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.parser import langchain_docs_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)


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

# def extract_items_from_rss(feed_url: str):
#     response = requests.get(feed_url)
#     soup = BeautifulSoup(response.content, "xml")
#     items = soup.find_all("item")
#     news_items = []

#     for item in items:
#         title = item.find("title").get_text() if item.find("title") else "No Title"
#         link = item.find("link").get_text() if item.find("link") else "No Link"
#         identifier = item.find("dc:identifier").get_text() if item.find("dc:identifier") else "No Identifier"
#         pub_date = item.find("pubDate").get_text() if item.find("pubDate") else "No Date"
#         creator = item.find("dc:creator").get_text() if item.find("dc:creator") else "No Creator"
#         thumbnail = item.find("media:thumbnail")['url'] if item.find("media:thumbnail") else "No Thumbnail"
#         guid = item.find("guid").get_text() if item.find("guid") else "No GUID"
#         description = item.find("description").get_text() if item.find("description") else "No Description"
#         content_encoded = item.find("content:encoded").decode_contents() if item.find("content:encoded") else "No Content"

#         news_item = {
#             "title": title,
#             "link": link,
#             "identifier": identifier,
#             "pub_date": pub_date,
#             "creator": creator,
#             "thumbnail": thumbnail,
#             "guid": guid,
#             "description": description,
#             "content": content_encoded,
#         }
#         news_items.append(news_item)


#     return news_items


# def load_sample_news():
#     feed_url = "https://cdn.feedcontrol.net/7512/12213-hIFHBiLc7Wh50.xml"
#     items = extract_items_from_rss(feed_url)
#     return items

# load_sample_news()

class RSSLoader:
    def __init__(self, feed_url: str):
        self.feed_url = feed_url
        self.news_items = []

    def fetch(self):
        # 获取RSS feed的内容
        response = requests.get(self.feed_url)
        self.soup = BeautifulSoup(response.content, "xml")
        return self

    def parse(self):
        # 找到所有的item元素
        items = self.soup.find_all("item")
        
        # 逐个item处理
        for item in items:
            # 对每个需要提取的字段做一个查找，如果不存在则用默认值
            title = item.find("title").get_text() if item.find("title") else "No Title"
            link = item.find("link").get_text() if item.find("link") else "No Link"
            identifier = item.find("dc:identifier").get_text() if item.find("dc:identifier") else "No Identifier"
            pub_date = item.find("pubDate").get_text() if item.find("pubDate") else "No Date"
            creator = item.find("dc:creator").get_text() if item.find("dc:creator") else "No Creator"
            thumbnail = item.find("media:thumbnail")['url'] if item.find("media:thumbnail") else "No Thumbnail"
            guid = item.find("guid").get_text() if item.find("guid") else "No GUID"
            description = item.find("description").get_text() if item.find("description") else "No Description"
            content_encoded = item.find("content:encoded").get_text() if item.find("content:encoded") else "No Content"

            news_item = {
                "title": title,
                "link": link,
                "identifier": identifier,
                "pub_date": pub_date,
                "creator": creator,
                "thumbnail": thumbnail,
                "guid": guid,
                "description": description,
                "content": content_encoded,
            }
            
            self.news_items.append(news_item)
        return self

    def load(self):
        return self.news_items

# 使用示例
def load_sample_news():
    return RSSLoader("https://cdn.feedcontrol.net/7512/12213-hIFHBiLc7Wh50.xml").fetch().parse().load()


def ingest_docs():
    WEAVIATE_URL = os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
    RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

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

        record_manager = SQLRecordManager(
            f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
        )
        record_manager.create_schema()

        docs_from_documentation = load_langchain_docs()
        logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
        docs_from_api = load_api_docs()
        logger.info(f"Loaded {len(docs_from_api)} docs from API")
        docs_from_langsmith = load_langsmith_docs()
        logger.info(f"Loaded {len(docs_from_langsmith)} docs from LangSmith")
        docs_from_langgraph = load_langgraph_docs()
        logger.info(f"Loaded {len(docs_from_langgraph)} docs from LangGraph")
        docs_from_sample_news = load_sample_news()
        logger.info(f"Loaded {len(docs_from_sample_news)} docs from SampleNews")

        docs_transformed = text_splitter.split_documents(
            docs_from_documentation
            + docs_from_api
            + docs_from_langsmith
            + docs_from_langgraph
            + docs_from_sample_news
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
        logger.info(
            f"LangChain now has this many vectors: {num_vecs}",
        )


if __name__ == "__main__":
    ingest_docs()
