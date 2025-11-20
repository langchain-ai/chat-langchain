"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
    get_weaviate_client: Create a Weaviate client connection.
"""

import os
import uuid
from contextlib import contextmanager
from typing import Any, Iterator, Literal, Optional, Union

import weaviate
from weaviate.auth import AuthApiKey
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


@contextmanager
def get_weaviate_client(
    weaviate_url: Optional[str] = None,
    weaviate_grpc_url: Optional[str] = None,
    weaviate_api_key: Optional[str] = None,
    http_port: int = 443,
    grpc_port: int = 443,
) -> Iterator[weaviate.WeaviateClient]:
    """Create and manage a Weaviate client connection.

    This is a context manager that creates a Weaviate client connection and ensures
    it is properly closed after use.

    Args:
        weaviate_url: The Weaviate HTTP URL. If None, reads from WEAVIATE_URL env var.
        weaviate_grpc_url: The Weaviate gRPC URL. If None, uses weaviate_url for both HTTP and gRPC.
        weaviate_api_key: The Weaviate API key. If None, reads from WEAVIATE_API_KEY env var.
        http_port: The HTTP port to use (default: 443).
        grpc_port: The gRPC port to use (default: 443).

    Yields:
        weaviate.WeaviateClient: A connected Weaviate client.

    Example:
        >>> with get_weaviate_client() as client:
        ...     # Use the client
        ...     collection = client.collections.get("MyCollection")
    """
    # Get URL and API key from environment if not provided
    url = weaviate_url or os.environ.get(
        "WEAVIATE_URL", "https://weaviate.hanu-nus.com"
    )
    api_key = weaviate_api_key or os.environ.get("WEAVIATE_API_KEY", "admin-key")

    # Extract hostname from URL (remove https:// or http://)
    http_host = url.replace("https://", "").replace("http://", "")

    # Use separate gRPC URL if provided, otherwise use the same as HTTP
    if weaviate_grpc_url:
        grpc_host = weaviate_grpc_url.replace("https://", "").replace("http://", "")
    else:
        grpc_host = http_host

    # Create Weaviate client
    client = weaviate.connect_to_custom(
        http_host=http_host,
        http_port=http_port,
        http_secure=True,
        grpc_host=grpc_host,
        grpc_port=grpc_port,
        grpc_secure=True,
        auth_credentials=AuthApiKey(api_key=api_key),
    )

    try:
        yield client
    finally:
        # Ensure the client is closed
        client.close()


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name

    model_kwargs = {"temperature": 0, "stream_usage": True}
    if provider == "google_genai":
        model_kwargs["convert_system_message_to_human"] = True
    return init_chat_model(model, model_provider=provider, **model_kwargs)


def reduce_docs(
    existing: Optional[list[Document]],
    new: Union[
        list[Document],
        list[dict[str, Any]],
        list[str],
        str,
        Literal["delete"],
    ],
) -> list[Document]:
    """Reduce and process documents based on the input type.

    This function handles various input types and converts them into a sequence of Document objects.
    It also combines existing documents with the new one based on the document ID.

    Args:
        existing (Optional[Sequence[Document]]): The existing docs in the state, if any.
        new (Union[Sequence[Document], Sequence[dict[str, Any]], Sequence[str], str, Literal["delete"]]):
            The new input to process. Can be a sequence of Documents, dictionaries, strings, or a single string.
    """
    if new == "delete":
        return []

    existing_list = list(existing) if existing else []
    if isinstance(new, str):
        return existing_list + [
            Document(page_content=new, metadata={"uuid": str(uuid.uuid4())})
        ]

    new_list = []
    if isinstance(new, list):
        existing_ids = set(doc.metadata.get("uuid") for doc in existing_list)
        for item in new:
            if isinstance(item, str):
                item_id = str(uuid.uuid4())
                new_list.append(Document(page_content=item, metadata={"uuid": item_id}))
                existing_ids.add(item_id)

            elif isinstance(item, dict):
                metadata = item.get("metadata", {})
                item_id = metadata.get("uuid", str(uuid.uuid4()))

                if item_id not in existing_ids:
                    new_list.append(
                        Document(**item, metadata={**metadata, "uuid": item_id})
                    )
                    existing_ids.add(item_id)

            elif isinstance(item, Document):
                item_id = item.metadata.get("uuid")
                if item_id is None:
                    item_id = str(uuid.uuid4())
                    new_item = item.copy(deep=True)
                    new_item.metadata["uuid"] = item_id
                else:
                    new_item = item

                if item_id not in existing_ids:
                    new_list.append(new_item)
                    existing_ids.add(item_id)

    return existing_list + new_list
