"""Load html from files, clean up, split, ingest into Weaviate."""

import logging
import os
import re
import sqlite3
from typing import Optional
import requests
import json

import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import SitemapLoader
from langchain.indexes import SQLRecordManager, index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore

from backend.constants import WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME
from backend.embeddings import get_embeddings_model
from backend.parser import langchain_docs_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]


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
        filter_urls=["https://js.langchain.com/docs/"],
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
    ).load()


def ingest_general_guides_and_tutorials():
    langchain_python_docs = load_langchain_python_docs()
    langchain_js_docs = load_langchain_js_docs()
    aggregated_site_docs = load_aggregated_docs_site()
    return langchain_python_docs + langchain_js_docs + aggregated_site_docs

def ingest_docs():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    with weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=weaviate.classes.init.Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
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
        docs_transformed = text_splitter.split_documents(
            general_guides_and_tutorials_docs
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

#########################
# API and SDK Docs
#########################

SQLITE_DB_PATH = "api_sdk_docs.db"

def create_sqlite_db():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS docs USING fts5(
        domain,
        title,
        url,
        content
    );
    """)
    conn.commit()
    return conn

def load_langsmith_api_docs():
    url = "https://api.smith.langchain.com/openapi.json"
    spec = requests.get(url).json()
    docs = parse_openapi_spec(spec, base_url="https://api.smith.langchain.com/redoc", domain="langsmith_api")
    return docs

def load_langgraph_platform_api_docs():
    with open("docs/langgraph-platform.json", "r") as f:
        spec = json.load(f)
    docs = parse_openapi_spec(spec, base_url="https://langchain-ai.github.io/langgraph/cloud/reference/api/api_ref.html", domain="langgraph_platform_api")
    return docs

def ingest_sdk_and_api_docs():
    conn = create_sqlite_db()
    cur = conn.cursor()
    docs = []
    docs.extend(load_langsmith_api_docs())
    docs.extend(load_langgraph_platform_api_docs())
    
    cur.execute("DELETE FROM docs WHERE domain = 'langsmith_api'")
    
    for doc in docs:
        cur.execute("""
        INSERT INTO docs (domain, title, url, content)
        VALUES (?, ?, ?, ?)
        """, (doc["domain"], doc["title"], doc["url"], doc["content"]))
    
    conn.commit()
    print(f"Inserted {len(docs)} documents into SQLite database")    
    conn.close()
    return docs

#########################
# Main
#########################

if __name__ == "__main__":
    ingest_docs()
    ingest_sdk_and_api_docs()

#########################
# Utils
#########################

def parse_openapi_spec(spec: dict, base_url: str, domain: str) -> list[dict]:
    docs = []
    for path, path_item in spec.get("paths", {}).items():
        for method, operation in path_item.items():
            if not isinstance(operation, dict):
                continue
            title = f"{method.upper()} {path}"
            if operation.get("summary"):
                title = f"{operation['summary']} ({method.upper()} {path})"
            
            content_parts = []
            
            if operation.get("operationId"):
                content_parts.append(f"Operation ID: {operation['operationId']}")
            
            if operation.get("description"):
                content_parts.append(f"Description: {operation['description']}")
            
            if operation.get("tags"):
                content_parts.append(f"Tags: {', '.join(operation['tags'])}")
            
            if operation.get("parameters"):
                param_info = []
                for param in operation["parameters"]:
                    param_str = f"- {param.get('name', 'unnamed')} ({param.get('in', 'unknown')})"
                    if param.get("required"):
                        param_str += " [required]"
                    if param.get("description"):
                        param_str += f": {param['description']}"
                    if param.get("schema"):
                        schema = param["schema"]
                        if "type" in schema:
                            param_str += f" (type: {schema['type']}"
                            if "format" in schema:
                                param_str += f", format: {schema['format']}"
                            param_str += ")"
                    param_info.append(param_str)
                
                if param_info:
                    content_parts.append("Parameters:\n" + "\n".join(param_info))
            
            if operation.get("requestBody"):
                req_body = operation["requestBody"]
                body_info = ["Request Body:"]
                if req_body.get("required"):
                    body_info.append("- Required: true")
                if req_body.get("description"):
                    body_info.append(f"- Description: {req_body['description']}")
                if req_body.get("content"):
                    content_types = list(req_body["content"].keys())
                    body_info.append(f"- Content-Types: {', '.join(content_types)}")
                    for content_type, content_spec in req_body["content"].items():
                        if content_spec.get("schema", {}).get("$ref"):
                            ref = content_spec["schema"]["$ref"]
                            schema_name = ref.split("/")[-1] if "/" in ref else ref
                            body_info.append(f"- Schema: {schema_name}")
                content_parts.append("\n".join(body_info))
            
            if operation.get("responses"):
                response_info = ["Responses:"]
                for status_code, response in operation["responses"].items():
                    resp_str = f"- {status_code}"
                    if response.get("description"):
                        resp_str += f": {response['description']}"
                    response_info.append(resp_str)
                content_parts.append("\n".join(response_info))
            
            if operation.get("security"):
                security_info = ["Security Requirements:"]
                for security_req in operation["security"]:
                    for scheme_name, scopes in security_req.items():
                        security_info.append(f"- {scheme_name}")
                        if scopes:
                            security_info.append(f"  Scopes: {', '.join(scopes)}")
                content_parts.append("\n".join(security_info))
            
            doc = {
                "domain": domain,
                "title": title,
                "url": base_url,
                "content": "\n\n".join(content_parts)
            }
            docs.append(doc)
    
    if "components" in spec and "schemas" in spec["components"]:
        for schema_name, schema_def in spec["components"]["schemas"].items():
            content_parts = [f"Schema: {schema_name}"]
            
            if schema_def.get("type"):
                content_parts.append(f"Type: {schema_def['type']}")
            
            if schema_def.get("description"):
                content_parts.append(f"Description: {schema_def['description']}")
            
            if schema_def.get("properties"):
                prop_info = ["Properties:"]
                for prop_name, prop_def in schema_def["properties"].items():
                    prop_str = f"- {prop_name}"
                    if prop_def.get("type"):
                        prop_str += f" ({prop_def['type']}"
                        if prop_def.get("format"):
                            prop_str += f", format: {prop_def['format']}"
                        prop_str += ")"
                    if prop_def.get("description"):
                        prop_str += f": {prop_def['description']}"
                    prop_info.append(prop_str)
                content_parts.append("\n".join(prop_info))
            
            if schema_def.get("required"):
                content_parts.append(f"Required fields: {', '.join(schema_def['required'])}")
            
            if schema_def.get("enum"):
                content_parts.append(f"Enum values: {', '.join(map(str, schema_def['enum']))}")
            
            doc = {
                "domain": domain,
                "title": f"Schema: {schema_name}",
                "url": f"{base_url}/schemas#{schema_name}",
                "content": "\n\n".join(content_parts)
            }
            docs.append(doc)
    return docs