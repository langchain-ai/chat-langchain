# Authentication and authorization for LangGraph deployment
import os

from langgraph_sdk import Auth
from langgraph_sdk.auth import is_studio_user

from src.agent.config import DEFAULT_MODEL, MODELS, PUBLIC_MODEL_IDS
from src.utils.prompt_provenance import get_prompt_provenance

auth = Auth()

MAX_RECURSION_LIMIT = 100
MAX_MESSAGE_CHARS = 50_000
IMAGE_UNSUPPORTED_MODEL_IDS = {MODELS["glm-5"].id}
UNSUPPORTED_IMAGE_MODEL_MESSAGE = "Selected model does not support image uploads"


def _get_auth_secret() -> str | None:
    """Optional auth secret. If set, X-Auth-Key header is required on all requests."""
    return os.getenv("LANGGRAPH_AUTH_SECRET")


@auth.authenticate
async def authenticate(
    authorization: str | None, headers: dict
) -> Auth.types.MinimalUserDict:
    """Validate requests and extract user identity.

    If LANGGRAPH_AUTH_SECRET is set, requires X-Auth-Key header to match.
    If not set, allows public requests through.

    User identity is always extracted from Authorization: Bearer <user_id>.
    """
    # If auth secret is configured, validate X-Auth-Key header
    auth_secret = _get_auth_secret()
    if auth_secret:
        auth_key = headers.get(b"x-auth-key") or headers.get("x-auth-key")
        if not auth_key:
            raise Auth.exceptions.HTTPException(
                status_code=401, detail="Authentication required"
            )
        key = auth_key.decode() if isinstance(auth_key, bytes) else auth_key
        if key != auth_secret:
            raise Auth.exceptions.HTTPException(
                status_code=401, detail="Invalid auth key"
            )

    # Extract user identity from Authorization header
    if not authorization:
        return {"identity": "studio-user"}

    user_id = authorization
    if authorization.lower().startswith("bearer "):
        user_id = authorization.split(" ", 1)[1]

    return {"identity": user_id or "anonymous", "is_authenticated": True}


# Default block
@auth.on
async def block_all(ctx: Auth.types.AuthContext, value: dict):
    if is_studio_user(ctx.user):
        return {}
    raise Auth.exceptions.HTTPException(403, "No access permitted")


@auth.on.threads
async def add_owner(ctx: Auth.types.AuthContext, value: dict):
    """Tag threads with their owner and restrict access."""
    if is_studio_user(ctx.user):
        return {}

    user_id = ctx.user.identity
    metadata = value.setdefault("metadata", {})
    metadata["user_id"] = user_id

    return {"user_id": user_id}


@auth.on.threads.update
async def update_owner_metadata(ctx: Auth.types.AuthContext, value: dict):
    """Allow users to update metadata only on their own threads."""
    if is_studio_user(ctx.user):
        return {}

    user_id = ctx.user.identity
    metadata = value.setdefault("metadata", {})
    metadata["user_id"] = user_id

    return {"user_id": user_id}


@auth.on.threads.create_run
async def enrich_run_metadata(
    ctx: Auth.types.AuthContext, value: Auth.types.RunsCreate
):
    """Inject public Chat LangChain metadata into the root run."""
    metadata = value.setdefault("metadata", {})
    if (
        value["assistant_id"] != "docs_agent"
        and str(value["assistant_id"]) != "bd5caeca-2e94-56a2-abb7-20aa1c78d5c8"
    ):
        raise Auth.exceptions.HTTPException(
            403,
            f"Only docs_agent runs are allowed to set source_type. Got {value['assistant_id']}",
        )

    config = value["kwargs"].get("config") or value.get("config") or {}
    config_metadata = config.get("metadata") if isinstance(config, dict) else None
    if isinstance(config_metadata, dict):
        config_source_type = config_metadata.get("source_type")
        if isinstance(config_source_type, str) and config_source_type:
            metadata.setdefault("source_type", config_source_type)

    metadata.setdefault("source_type", "Chat-LangChain")

    graph_id = metadata.get("graph_id") or value.get("assistant_id")
    if graph_id:
        for key, val in get_prompt_provenance(graph_id).items():
            metadata.setdefault(key, val)
    input_has_image = validate_inputs(
        value["kwargs"].get("input"), value["kwargs"].get("command")
    )
    validate_config(
        value["kwargs"].get("config") or value.get("config"),
        input_has_image=input_has_image,
    )


@auth.on.assistants(actions=["create", "update", "delete"])
async def block_modify_assistants(
    ctx: Auth.types.AuthContext, value: Auth.types.AssistantsCreate
):
    if is_studio_user(ctx.user):
        return {}
    raise Auth.exceptions.HTTPException(403, "Modifying assistants is not allowed")


@auth.on.assistants.read
async def block_modify_assistants(
    ctx: Auth.types.AuthContext, value: Auth.types.AssistantsRead
):
    if is_studio_user(ctx.user):
        return {}

    user_id = ctx.user.identity
    metadata = value.setdefault("metadata", {})
    metadata["user_id"] = user_id

    return {"user_id": user_id}


def validate_inputs(input: dict | None, command: dict | None) -> bool:
    if command:
        raise Auth.exceptions.HTTPException(422, "Command not accepted")
    if input is None:
        raise Auth.exceptions.HTTPException(422, "Input is required")
    if not isinstance(input, dict):
        raise Auth.exceptions.HTTPException(
            422, f"Unrecognized input: {type(input)}"
        )
    if not input:
        raise Auth.exceptions.HTTPException(422, "Input is required")

    messages = input.get("messages")
    if messages is None:
        raise Auth.exceptions.HTTPException(422, "Messages are required")
    if isinstance(messages, str):
        if not messages.strip():
            raise Auth.exceptions.HTTPException(422, "Message content is required")
        input["messages"] = messages[:MAX_MESSAGE_CHARS]
        return False
    if not isinstance(messages, list):
        raise Auth.exceptions.HTTPException(
            422, f"Unrecognized messages input: {type(messages)}"
        )
    if len(messages) != 1:
        raise Auth.exceptions.HTTPException(
            422, f"Only one message accepted per term. Got {len(messages)}"
        )
    msg = messages[0]
    if isinstance(msg, str):
        if not msg.strip():
            raise Auth.exceptions.HTTPException(422, "Message content is required")
        messages[0] = msg[:MAX_MESSAGE_CHARS]
        return False
    if not isinstance(msg, dict):
        raise Auth.exceptions.HTTPException(
            422, f"Unrecognized message input: {type(msg)}"
        )
    role = msg.get("role") or msg.get("type")
    if role not in ("user", "human"):
        raise Auth.exceptions.HTTPException(
            422, f"Only user messages accepted. Got role {role}"
        )
    content = msg.get("content")
    if content is None:
        raise Auth.exceptions.HTTPException(422, "Message content is required")
    if isinstance(content, str) and not content.strip():
        raise Auth.exceptions.HTTPException(422, "Message content is required")
    if isinstance(content, list) and not content:
        raise Auth.exceptions.HTTPException(422, "Message content is required")
    msg["content"] = truncate_message_content(content)
    return content_has_image(msg["content"])


def truncate_message_content(content):
    """Trim user-provided text while preserving non-text content blocks."""
    if isinstance(content, str):
        return content[:MAX_MESSAGE_CHARS]

    if not isinstance(content, list):
        return content

    remaining = MAX_MESSAGE_CHARS
    truncated = []
    for block in content:
        if isinstance(block, str):
            text = block[:remaining]
            truncated.append(text)
            remaining -= len(text)
            continue

        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            text = block["text"][:remaining]
            truncated.append({**block, "text": text})
            remaining -= len(text)
            continue

        truncated.append(block)

    return truncated


def content_has_image(content) -> bool:
    """Return whether message content contains an image block."""
    if not isinstance(content, list):
        return False

    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")
        if block_type in ("image", "image_url"):
            return True

        mime_type = block.get("mime_type") or block.get("mimeType")
        if isinstance(mime_type, str) and mime_type.startswith("image/"):
            return True

        if "image_url" in block:
            return True

    return False


def validate_config(config: dict | None, *, input_has_image: bool = False):
    """Validate user-controlled run config before it reaches the graph."""
    if not config:
        return
    if not isinstance(config, dict):
        raise Auth.exceptions.HTTPException(
            422, f"Unrecognized config input: {type(config)}"
        )

    cap_recursion_limit(config)

    configurable = config.get("configurable") or {}
    if not isinstance(configurable, dict):
        raise Auth.exceptions.HTTPException(
            422, f"Unrecognized configurable input: {type(configurable)}"
        )

    requested_model = configurable.get("model")
    if requested_model is None:
        return
    if not isinstance(requested_model, str):
        raise Auth.exceptions.HTTPException(
            422, f"Unrecognized model input: {type(requested_model)}"
        )

    if requested_model == DEFAULT_MODEL.id:
        return

    if requested_model not in PUBLIC_MODEL_IDS:
        raise Auth.exceptions.HTTPException(
            422, f"Model is not allowed: {requested_model}"
        )

    if input_has_image and requested_model in IMAGE_UNSUPPORTED_MODEL_IDS:
        raise Auth.exceptions.HTTPException(422, UNSUPPORTED_IMAGE_MODEL_MESSAGE)


def cap_recursion_limit(config: dict):
    recursion_limit = config.get("recursion_limit")
    if recursion_limit is None:
        return

    if isinstance(recursion_limit, bool) or not isinstance(recursion_limit, int):
        raise Auth.exceptions.HTTPException(
            422, "recursion_limit must be an integer"
        )

    if recursion_limit > MAX_RECURSION_LIMIT:
        config["recursion_limit"] = MAX_RECURSION_LIMIT
