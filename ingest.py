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
from langchain.storage import RedisStore, EncoderBackedStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.utilities.redis import get_client
import json

WEAVIATE_URL=os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY=os.environ["WEAVIATE_API_KEY"]

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

    python_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                chunk_size=2000, 
                                                                chunk_overlap=200)
    texts = python_splitter.split_documents(documents_repo)
    return texts

def key_encoder(key: int) -> str:
    return json.dumps(key)

def value_serializer(value: float) -> str:
    if isinstance(value, Document):
        value = {
            'page_content': value.page_content,
            'metadata': value.metadata,
        }
    return json.dumps(value)

def value_deserializer(serialized_value: str) -> Document:
    value = json.loads(serialized_value)
    if 'page_content' in value and 'metadata' in value:
        return Document(page_content=value['page_content'], metadata=value['metadata'])
    else:
        return value
    
def ingest_docs():
    """Get documents from web pages."""

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
        loader = RecursiveUrlLoader(url=url, max_depth=2 if url == urls[0] else 10, extractor=lambda x: Soup(x, "lxml").text, prevent_outside=True)
        temp_docs = loader.load()
        temp_docs = [doc for i, doc in enumerate(temp_docs) if doc not in temp_docs[:i]]        
        documents += temp_docs
        print("Loaded", len(temp_docs), "documents from", url)
    
    print("Loaded", len(documents), "documents from all URLs")
    
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(documents)
    
    repo_docs = ingest_repo()
    docs_transformed += repo_docs
    
    print("Transformed", len(docs_transformed), "documents in total")
    
    # OPTION TO PICKLE
    # import pickle
    # with open('docs_transformed.pkl', 'wb') as f:
    #     pickle.dump(docs_transformed, f)
    # with open('docs_transformed.pkl', 'rb') as f:
    #     docs_transformed = pickle.load(f)
    
    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    client.schema.delete_class("LangChain_parents_idx") # delete the class if it already exists
    
    vectorstore = Weaviate(
        embedding=OpenAIEmbeddings(chunk_size=200),
        client=client,
        index_name="LangChain_parents_idx",
        by_text=False,
        text_key="text",
        attributes=["doc_id"],
    )

    redis_client = get_client('redis://localhost:6379')
    abstract_store = RedisStore(client=redis_client)

    # Create an instance of the encoder-backed store
    store = EncoderBackedStore(
        store=abstract_store,
        key_encoder=key_encoder,
        value_serializer=value_serializer,
        value_deserializer=value_deserializer
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore, 
        docstore=store, 
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )
            
    # client.schema.delete_all()
    # print(client.schema.get()) # should be empty
        
    batch = 100
    for i in range(0, len(docs_transformed), batch):
        chunk = docs_transformed[i:i+batch]
        retriever.add_documents(chunk, ids=None)

    print("LangChain now has this many vectors", client.query.aggregate("LangChain_parents_idx").with_meta_count().do())
    
if __name__ == "__main__":
    ingest_docs()
    