from typing import List, Optional

from dotenv import load_dotenv
from weaviate.collections.classes.internal import QueryReturn

from croptalk.weaviate_utils import (
    get_client_collection,
    query_near_text_with_filters,
)

load_dotenv("secrets/.env.secret")
load_dotenv("secrets/.env.shared")


class DocumentRetriever:
    """
    Class responsible for document retrieval in a weaviate vector store.
    """

    def __init__(self, collection_name: str) -> None:
        """
        Connecting to weaviate cloud services requires the following environment variables to be
        set:
            - WCS_CLUSTER_URL
            - WCS_API_KEY

        Args:
            collection_name: Name of weaviate collection
        """
        self.collection_name = collection_name
        self.collection = get_client_collection(self.collection_name)[1]

    def get_documents(
        self,
        query: str,
        doc_category: Optional[str] = None,
        commodity: Optional[str] = None,
        county: Optional[str] = None,
        state: Optional[str] = None,
        top_k: int = 3,
        include_common_docs: bool = True,
    ) -> List[str]:
        """
        Args:
            query: query to use for document retrieval
            doc_category: document category to filter on, None means no filter
            commodity: commodity to filter on, None means no filter
            county: county to filter on, None means no filter
            state: state to filter on, None means no filter
            top_k: number of retrieved documents we are aiming for, defaults to 3
            include_common_docs: whether (default) or not to include documents that apply to all
                                 states, counties or commodities... does not apply to document
                                 category filter

        Returns:
            list of retrieved documents matching query, filters and top_k
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string. Received: {query}")

        # query vector store
        query_response = query_near_text_with_filters(
            collection=self.collection,
            query=query,
            limit=top_k,
            doc_category=doc_category,
            commodity=commodity,
            county=county,
            state=state,
            include_common_docs=include_common_docs,
        )

        # format returned docs
        formatted_docs = self._format_query_response(query_response)
        return formatted_docs

    @staticmethod
    def _format_query_response(query_response: QueryReturn) -> List[str]:
        """
        Args:
            query_response: weaviate query response

        Returns:
            list of retrieved and formatted documents, equivalent to provided query response
        """
        return [
            f"<doc"
            f" id='{i+1}'"
            f" title='{doc.properties['title']}'"
            f" page_id='{doc.properties['page_start']}'"
            f" doc_category='{doc.properties['doc_category']}'"
            f" commodity='{doc.properties['commodity']}'"
            f" state='{doc.properties['state']}'"
            f" county='{doc.properties['county']}'"
            f" s3_key='{doc.properties['s3_key']}'"
            f" url='https://croptalk-spoi.s3.us-east-2.amazonaws.com/{doc.properties['s3_key']}'"
            f">{doc.properties['content']}</doc>"
            for i, doc in enumerate(query_response.objects)
        ]
