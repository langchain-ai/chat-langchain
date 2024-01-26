from typing import Callable, List, Optional

from chromadb.api.types import QueryResult
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from croptalk.chromadb_utils import create_chroma_filter, get_chroma_collection


class DocumentRetriever:
    """
    Class responsible for document retrieval in a ChromaDB vector store.
    """

    def __init__(
        self,
        vectorestore_dir: str,
        collection_name: str,
        embedding_function: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            vectorestore_dir: directory where ChromaDB vectorstore files are located
            collection_name: collection name
            embedding_function: embedding function to use,
                                ChromaDB's default embedding function will be used if none is
                                provided
        """
        self.vectorestore_dir = vectorestore_dir
        self.collection_name = collection_name
        self.embedding_function = embedding_function or DefaultEmbeddingFunction()

        self.collection = get_chroma_collection(
            vectorestore_dir=self.vectorestore_dir,
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
        )

    def get_documents(
        self,
        query: str,
        doc_category: Optional[str] = None,
        commodity: Optional[str] = None,
        county: Optional[str] = None,
        state: Optional[str] = None,
        top_k: int = 3,
    ) -> List[str]:
        """
        Args:
            query: query to use for document retrieval
            doc_category: document category to filter on
            commodity: commodity to filter on
            county: county to filter on
            state: state to filter on
            top_k: number of retrieved documents we are aiming for, defaults to 3

        Returns:
            list of retrieved documents matching query and filters
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string. Received: {query}")

        query_embedding = self.embedding_function([query])
        where_filter = create_chroma_filter(
            commodity=commodity,
            county=county,
            state=state,
            doc_category=doc_category,
            include_common_docs=True,
        )
        result = self.collection.query(query_embedding, where=where_filter, n_results=top_k)
        formatted_docs = self._format_docs(result)
        return formatted_docs

    @staticmethod
    def _format_docs(result: QueryResult) -> List[str]:
        """
        Args:
            result: ChromaDB query result

        Returns:
            list of retrieved and formatted documents, equivalent to provided query result
        """
        doc_contents = result["documents"][0]
        doc_metadatas = result["metadatas"][0]
        return [
            f"<doc"
            f" id='{i+1}',"
            f" title={metadata['title']},"
            f" page_id={metadata['page']},"
            f" doc_category={metadata['doc_category']},"
            f" url={metadata['source']}"
            f">{content}</doc>"
            for i, (content, metadata)
            in enumerate(zip(doc_contents, doc_metadatas))
        ]
