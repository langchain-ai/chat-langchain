"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""

import uuid
from typing import Any, Literal, Optional, Union

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


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

    model_kwargs = {"temperature": 0}
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
            
            if domain == "langsmith_api" and operation.get("tags") and operation.get("operationId"):
                tag = operation["tags"][0] if operation["tags"] else ""
                operation_id = operation["operationId"]
                specific_url = f"{base_url}#tag/{tag}/operation/{operation_id}"
            elif domain == "langgraph_platform_api" and operation.get("tags"):
                # Handle tag formatting: "Crons (Plus Tier)" -> "crons-plus-tier"
                tag = operation["tags"][0] if operation["tags"] else ""
                tag = tag.lower().replace(" (", "-").replace(")", "").replace(" ", "-")
                path_without_leading_slash = path.lstrip("/")
                specific_url = f"{base_url}#tag/{tag}/{method.lower()}/{path_without_leading_slash}"
            else:
                specific_url = base_url
            
            doc = {
                "domain": domain,
                "title": title,
                "url": specific_url,
                "content": "\n\n".join(content_parts)
            }
            docs.append(doc)
            if domain == "langgraph_platform_api":
                print(doc)
    
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
                "url": base_url,
                "content": "\n\n".join(content_parts)
            }
            docs.append(doc)
    return docs