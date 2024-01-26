import os
import chromadb
from typing import List, Optional, Sequence

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document

from croptalk.chromadb_utils import create_chroma_filter

from dotenv import load_dotenv
load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')


def get_retriever(vectorestore_dir: str, collection_name: str, k: int = 3) -> BaseRetriever:
    """Creates a langchain version of chromadb retriever that can additionaly filter documents by 'filter' argument"""

    # Connect to the existing collection through native Chroma's API
    chroma_client = chromadb.PersistentClient(path=vectorestore_dir)

    # Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model to create embeddings by default
    # So if we are to use native chromabd retriever, we can specify embedding function as follows:
    # from chromadb.utils import embedding_functions
    # emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    # However here we have to use langchain's SentenceTransformerEmbeddings wrapper
    lanngchain_emb = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2")

    # Load the collection through langchain's Chroma wrapper
    vectorstore = Chroma(
        client=chroma_client,
        collection_name=collection_name,
        embedding_function=lanngchain_emb,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Here we are overriding the _get_relevant_documents method of the retriever
    # to use the similarity search method with the filter argument
    def _get_relevant_documents(
        self, query: str, run_manage=None, filter=None, **kwargs
    ) -> List[Document]:
        if self.search_type == "similarity":
            docs = self.vectorstore.similarity_search(
                query, filter=filter, **self.search_kwargs)
        return docs

    VectorStoreRetriever._get_relevant_documents = _get_relevant_documents
    VectorStoreRetriever._expects_other_args = True

    return retriever


def retriever_with_filter(query: str, doc_category: str = None,
                          commodity: str = None, county: str = None, state: str = None, **kwargs) -> List[Document]:
    """Retriever wrapper that allows to create chromadb where_filter and filter documents by there metadata."""

    where_filter = create_chroma_filter(commodity=commodity, county=county, state=state,
                                        doc_category=doc_category, include_common_docs=False)

    return retriever.get_relevant_documents(query, filter=where_filter)


class RetrieverInput(BaseModel):
    """Input schema for an llm-toolkit retriever."""
    query: str = Field(description="Query to look up in retriever")
    commodity: Optional[str] = Field(
        description="Commodity name. Example: Apples")
    state: Optional[str] = Field(description="State name. Example: California")
    county: Optional[str] = Field(description="County name. Example: Ventura")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i+1}' page_id={doc.metadata['page']} doc_category={doc.metadata['doc_category']}>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return formatted_docs


def retriever_with_filter_each_category(query: str, commodity: str = None, county: str = None, state: str = None,
                                        formatted=True, **kwargs) -> List[Document]:
    """Retriever wrapper that allows to create chromadb where_filter and filter documents by there metadata."""
    # TODO: run requests in async manner
    cih_docs = retriever.get_relevant_documents(
        query, filter=create_chroma_filter(doc_category='CIH'))
    bp_docs = retriever.get_relevant_documents(
        query, filter=create_chroma_filter(doc_category='BP'))
    cp_docs = retriever.get_relevant_documents(
        query, filter=create_chroma_filter(doc_category='CP', commodity=commodity))
    sp_docs = retriever.get_relevant_documents(query, filter=create_chroma_filter(doc_category='SP',
                                                                                  commodity=commodity, state=state, county=county))

    docs = cih_docs + bp_docs + cp_docs + sp_docs
    if formatted:
        return "\n".join(format_docs(docs))
    return docs


vectorestore_dir = os.getenv("VECTORSTORE_DIR")
collection = os.getenv("VECTORSTORE_COLLECTION")
top_k = int(os.getenv("VECTORSTORE_TOP_K"))

retriever = get_retriever(
    vectorestore_dir, collection_name=collection, k=top_k)
