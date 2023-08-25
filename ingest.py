"""Load html from files, clean up, split, ingest into Weaviate."""
import json
import os
import shutil
import uuid

import weaviate
from bs4 import BeautifulSoup as Soup
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.schema import Document
from langchain.storage import EncoderBackedStore, RedisStore
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter
from langchain.utilities.redis import get_client
from langchain.vectorstores import Weaviate

WEAVIATE_URL = "http://localhost:8080"
WEAVIATE_API_KEY = "foo"


def _load_split_repo():
    repo_path = os.path.join(os.getcwd(), "test_repo")
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    loader = GenericLoader.from_filesystem(
        repo_path + "/libs/langchain/langchain",
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500),
    )
    documents_repo = loader.load()
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
    )
    return python_splitter.split_documents(documents_repo)


def _load_split_docs():
    """Get documents from web pages."""

    urls = [
        "https://api.python.langchain.com/en/latest/api_reference.html#module-langchain",
        "https://python.langchain.com/docs/get_started",
        "https://python.langchain.com/docs/use_cases",
        # "https://python.langchain.com/docs/integrations",
        # "https://python.langchain.com/docs/modules",
        # "https://python.langchain.com/docs/guides",
        # "https://python.langchain.com/docs/ecosystem",
        # "https://python.langchain.com/docs/additional_resources",
        # "https://python.langchain.com/docs/community",
    ]

    documents = []
    for url in urls:
        loader = RecursiveUrlLoader(
            url=url,
            max_depth=10,
            extractor=lambda x: Soup(x, "lxml").text,
            prevent_outside=True,
        )
        temp_docs = loader.load()
        temp_docs = [
            doc
            for i, doc in enumerate(temp_docs)
            if doc not in temp_docs[:i] and doc not in documents
        ]
        documents += temp_docs
        print("Loaded", len(temp_docs), "documents from", url)
    print("Loaded", len(documents), "documents from all URLs")
    html2text = Html2TextTransformer()
    return list(html2text.transform_documents(documents))


def _value_serializer(value) -> str:
    if isinstance(value, Document):
        value = {
            "page_content": value.page_content,
            "metadata": value.metadata,
        }
    return json.dumps(value)


def _value_deserializer(serialized_value: str) -> Document:
    value = json.loads(serialized_value)
    if "page_content" in value and "metadata" in value:
        return Document(page_content=value["page_content"], metadata=value["metadata"])
    else:
        return value


def _index_docs(
    vectorstore,
    docstore,
    record_manager,
    parent_splitter,
    child_splitter,
    documents,
    id_key,
):
    if parent_splitter is not None:
        documents = parent_splitter.split_documents(documents)
    doc_ids = [str(uuid.uuid4()) for _ in documents]
    docs = []
    full_docs = []
    for i, doc in enumerate(documents):
        _id = doc_ids[i]
        sub_docs = child_splitter.split_documents([doc])
        for _doc in sub_docs:
            _doc.metadata[id_key] = _id
        docs.extend(sub_docs)
        full_docs.append((_id, doc))

    index(docs, record_manager, vectorstore, delete_mode="full", source_id_key="source")
    vectorstore.add_documents(docs)
    docstore.mset(full_docs)


def _ingest():
    docs = _load_split_docs() + _load_split_repo()
    print("Transformed", len(docs), "documents in total")

    client = weaviate.Client(
        url=WEAVIATE_URL,
    )

    vectorstore = Weaviate(
        embedding=OpenAIEmbeddings(chunk_size=200),
        client=client,
        index_name="LangChain_parents_idx",
        by_text=False,
        text_key="text",
        attributes=["doc_id"],
    )

    redis_client = get_client("redis://localhost:6379")
    abstract_store = RedisStore(client=redis_client, namespace="parent_docs")

    # clear doc store
    # abstract_store.mdelete(list(abstract_store.yield_keys()))

    # Create an instance of the encoder-backed store
    store = EncoderBackedStore(
        store=abstract_store,
        key_encoder=json.dumps,
        value_serializer=_value_serializer,
        value_deserializer=_value_deserializer,
    )

    namespace = f"weaviate/test_index"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    _index_docs(
        vectorstore,
        store,
        record_manager,
        parent_splitter,
        child_splitter,
        docs,
        "parent_id",
    )
    print(
        "LangChain now has this many vectors",
        client.query.aggregate("LangChain_parents_idx").with_meta_count().do(),
    )


if __name__ == "__main__":
    _ingest()
