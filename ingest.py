"""Load html from files, clean up, split, ingest into Weaviate."""
from bs4 import BeautifulSoup as Soup
import weaviate
import os
from git import Repo
import shutil

from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.document_transformers import Html2TextTransformer
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser

WEAVIATE_URL=os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY=os.environ["WEAVIATE_API_KEY"]

def ingest_repo():
    repo_path = "/Users/mollycantillon/Desktop/test_repo"
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

    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                chunk_size=1000, 
                                                                chunk_overlap=200)
    texts = python_splitter.split_documents(documents_repo)
    return texts

def ingest_docs():
    """Get documents from web pages."""

    urls = [
        "https://api.python.langchain.com/en/latest/api_reference.html#module-langchain",
        "https://python.langchain.com/docs/get_started", 
        "https://python.langchain.com/docs/modules", 
        "https://python.langchain.com/docs/guides",
        "https://python.langchain.com/docs/ecosystem",
        "https://python.langchain.com/docs/additional_resources",
    ]

    documents = []
    for url in urls:
        loader = RecursiveUrlLoader(url=url, max_depth=1 if url == urls[0] else 3, extractor=lambda x: Soup(x, "lxml").text, prevent_outside=False if url=urls[0] else True)
        documents += loader.load()
    
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    print('before splitting', len(docs_transformed))
    docs = text_splitter.split_documents(docs_transformed)
    print('after splitting', len(docs))
    
    repo_docs = ingest_repo()
    docs += repo_docs

    embeddings = OpenAIEmbeddings(chunk_size=200) # rate limit

    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))
    client.schema.delete_all()
    print(client.schema.get()) # should be empty

    batch_size = 100 # to handle batch size limit 
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        Weaviate.from_documents(batch, embeddings, client=client, by_text=False, index_name="LangChain_idx")

    print("LangChain now has this many vectors", client.query.aggregate("LangChain_idx").with_meta_count().do())
    
if __name__ == "__main__":
    ingest_docs()
    