"""Load html from files, clean up, split, ingest into Weaviate."""
import os
from git import Repo
import shutil
import pickle
from bs4 import BeautifulSoup
from typing import Optional
import weaviate
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import SQLRecordManager, index
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.vectorstores import Weaviate
from langchain.document_transformers import Html2TextTransformer
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.chat_models import ChatOpenAI
from langchain.agents import (
    Tool,
    AgentExecutor,
)
from langchain.prompts import MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory

WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
WEAVIATE_REPO_INDEX_NAME = "LangChain_agent_repo"
WEAVIATE_DOCS_INDEX_NAME = "LangChain_agent_docs"
WEAVIATE_SOURCES_INDEX_NAME = "LangChain_agent_sources"
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]


def ingest_repo():
    repo_path = os.path.join(os.getcwd(), "test_repo")
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)

    repo = Repo.clone_from("https://github.com/langchain-ai/langchain", to_path=repo_path)

    loader = GenericLoader.from_filesystem(
        repo_path+"/libs/langchain/langchain",
        glob="**/*",
        suffixes=[".py"],
        parser=LanguageParser(language=Language.PYTHON, parser_threshold=500)
    )
    documents_repo = loader.load()
    len(documents_repo)

    with open('agent_repo_transformed.pkl', 'wb') as f:
        pickle.dump(documents_repo, f)
        
    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                chunk_size=2000, 
                                                                chunk_overlap=200)
    texts = python_splitter.split_documents(documents_repo)
    
    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))
    embedding = OpenAIEmbeddings(chunk_size=200)
    vectorstore = Weaviate(
        client,
        WEAVIATE_REPO_INDEX_NAME,
        "text",
        embedding=embedding,
        by_text=False,
        attributes=["source"]
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_REPO_INDEX_NAME}",
        db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()
    index(
        texts,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source"
    )
    return texts


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
        loader = RecursiveUrlLoader(url=url, max_depth=8, extractor=lambda x: BeautifulSoup(x, "lxml").text, prevent_outside=True)
        temp_docs = loader.load()
        temp_docs = [doc for i, doc in enumerate(temp_docs) if doc not in temp_docs[:i]]        
        documents += temp_docs
    
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    with open('agent_docs_transformed.pkl', 'wb') as f:
        pickle.dump(docs_transformed, f)
        
    docs_transformed = text_splitter.split_documents(docs_transformed)
    
    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY)
    )
    embedding = OpenAIEmbeddings(chunk_size=200)  # rate limit
    vectorstore = Weaviate(
        client,
        WEAVIATE_DOCS_INDEX_NAME,
        "text",
        embedding=embedding,
        by_text=False,
        attributes=["source"]
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}",
        db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()
    index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source"
    )

    print(
        "LangChain now has this many vectors: ",
        client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
    )

def ingest_sources():
    with open('agent_repo_transformed.pkl', 'rb') as f:
        codes = pickle.load(f)

    with open('agent_docs_transformed.pkl', 'rb') as f:
        documentations = pickle.load(f)

    all_texts = codes + documentations
    with open('agent_all_transformed.pkl', 'wb') as f:
        pickle.dump(all_texts, f)
    
    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))
    embedding = OpenAIEmbeddings(chunk_size=200)  # rate limit
    vectorstore = Weaviate(
        client,
        WEAVIATE_SOURCES_INDEX_NAME,
        "text",
        embedding=embedding,
        by_text=False,
        attributes=["source"]
    )

    record_manager = SQLRecordManager(
        f"weaviate/{WEAVIATE_SOURCES_INDEX_NAME}",
        db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()
    index(
        all_texts,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source"
    )


if __name__ == "__main__":
    ingest_repo()
    ingest_docs()
    ingest_sources()
