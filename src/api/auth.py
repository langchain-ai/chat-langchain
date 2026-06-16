# Authentication and authorization for LangGraph deployment
import base64
import binascii
import hashlib
import hmac
import json
import os
import time
from typing import Any

import httpx
from langgraph_sdk import Auth
from langgraph_sdk.auth import is_studio_user

from src.utils.prompt_provenance import get_prompt_provenance

auth = Auth()

MAX_RECURSION_LIMIT = 100
MAX_MESSAGE_CHARS = 50_000
GUEST_TOKEN_PREFIX = "guest."
AUTH_REGIONS = ("us", "eu", "apac", "aws")
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("BACKEND_RATE_LIMIT_MAX_REQUESTS", "20"))
RATE_LIMIT_WINDOW_SECONDS = float(os.getenv("BACKEND_RATE_LIMIT_WINDOW_SECONDS", "60"))
RATE_LIMIT_EVICTION_INTERVAL_SECONDS = 5 * 60
_rate_limit_entries: dict[str, list[float]] = {}
_last_rate_limit_eviction = 0.0


def _get_user_field(user: Any, field: str) -> Any:
    if isinstance(user, dict):
        return user.get(field)
    return getattr(user, field, None)


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    if authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return authorization.strip()


def _header_to_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("latin-1")
    return str(value)


def _get_header(headers: dict, name: str) -> str | None:
    for key, value in headers.items():
        if _header_to_str(key).lower() == name:
            return _header_to_str(value)
    return None


def _get_client_ip(headers: dict) -> str:
    forwarded_for = _get_header(headers, "x-forwarded-for")
    if forwarded_for:
        ip = forwarded_for.split(",", 1)[0].strip()
        if ip:
            return ip[:128]

    real_ip = _get_header(headers, "x-real-ip")
    if real_ip:
        ip = real_ip.strip()
        if ip:
            return ip[:128]

    return "unknown"


def _evict_stale_rate_limit_entries(now: float) -> None:
    global _last_rate_limit_eviction
    if now - _last_rate_limit_eviction < RATE_LIMIT_EVICTION_INTERVAL_SECONDS:
        return

    _last_rate_limit_eviction = now
    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    for ip, timestamps in list(_rate_limit_entries.items()):
        fresh = [timestamp for timestamp in timestamps if timestamp > cutoff]
        if fresh:
            _rate_limit_entries[ip] = fresh
        else:
            del _rate_limit_entries[ip]


def _check_rate_limit(headers: dict, now: float | None = None) -> None:
    _check_rate_limit_for_ip(_get_client_ip(headers), now)


def _check_rate_limit_for_ip(ip: str, now: float | None = None) -> None:
    now = time.time() if now is None else now
    _evict_stale_rate_limit_entries(now)

    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    timestamps = [
        timestamp for timestamp in _rate_limit_entries.get(ip, []) if timestamp > cutoff
    ]

    if len(timestamps) >= RATE_LIMIT_MAX_REQUESTS:
        _rate_limit_entries[ip] = timestamps
        raise Auth.exceptions.HTTPException(
            status_code=429, detail="Too many requests"
        )

    timestamps.append(now)
    _rate_limit_entries[ip] = timestamps


def _reset_rate_limit_for_tests() -> None:
    global _last_rate_limit_eviction
    _rate_limit_entries.clear()
    _last_rate_limit_eviction = 0.0


def _base64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(f"{value}{padding}")


def _verify_guest_token(token: str) -> str | None:
    """Return the guest identity from a server-signed guest token."""
    secret = os.getenv("GUEST_AUTH_SECRET")
    if not secret or not token.startswith(GUEST_TOKEN_PREFIX):
        return None

    try:
        _, payload_b64, signature_b64 = token.split(".", 2)
    except ValueError:
        return None

    expected = hmac.new(
        secret.encode(),
        payload_b64.encode(),
        hashlib.sha256,
    ).digest()

    try:
        actual = _base64url_decode(signature_b64)
    except (binascii.Error, ValueError):
        return None

    if not hmac.compare_digest(expected, actual):
        return None

    try:
        payload = json.loads(_base64url_decode(payload_b64))
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError):
        return None

    if payload.get("typ") != "guest":
        return None
    if not isinstance(payload.get("exp"), (int, float)) or payload["exp"] < time.time():
        return None

    guest_id = payload.get("sub")
    if not isinstance(guest_id, str) or not guest_id.startswith("user-"):
        return None
    return guest_id


def _supabase_config_for_region(region: str) -> tuple[str | None, str | None]:
    if region == "eu":
        return (
            os.getenv("SUPABASE_EU_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_EU_URL"),
            os.getenv("SUPABASE_EU_ANON_KEY")
            or os.getenv("NEXT_PUBLIC_SUPABASE_EU_ANON_KEY"),
        )
    if region == "apac":
        return (
            os.getenv("SUPABASE_APAC_URL")
            or os.getenv("NEXT_PUBLIC_SUPABASE_APAC_URL"),
            os.getenv("SUPABASE_APAC_ANON_KEY")
            or os.getenv("NEXT_PUBLIC_SUPABASE_APAC_ANON_KEY"),
        )
    if region == "aws":
        return (
            os.getenv("SUPABASE_AWS_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_AWS_URL"),
            os.getenv("SUPABASE_AWS_ANON_KEY")
            or os.getenv("NEXT_PUBLIC_SUPABASE_AWS_ANON_KEY"),
        )
    return (
        os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL"),
        os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY"),
    )


def _configured_supabase_regions() -> list[tuple[str, str, str]]:
    configs = []
    for region in AUTH_REGIONS:
        url, anon_key = _supabase_config_for_region(region)
        if url and anon_key:
            configs.append((region, url, anon_key))
    return configs


async def _verify_supabase_token_for_project(
    token: str, supabase_url: str, supabase_anon_key: str
) -> str | None:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{supabase_url.rstrip('/')}/auth/v1/user",
                headers={
                    "Authorization": f"Bearer {token}",
                    "apikey": supabase_anon_key,
                },
            )
    except httpx.HTTPError:
        return None

    if response.status_code != 200:
        return None

    try:
        user: dict[str, Any] = response.json()
    except json.JSONDecodeError:
        return None

    email = user.get("email")
    user_id = user.get("id")
    if isinstance(email, str) and email:
        return email
    if isinstance(user_id, str) and user_id:
        return f"auth:{user_id}"
    return None


async def _verify_supabase_token(
    token: str, selected_region: str | None = None
) -> str | None:
    """Return the authenticated user email from a Supabase access token."""
    configs = _configured_supabase_regions()
    if not configs:
        return None

    if selected_region:
        selected_config = next(
            (config for config in configs if config[0] == selected_region),
            None,
        )
        if not selected_config:
            return None
        _, supabase_url, supabase_anon_key = selected_config
        return await _verify_supabase_token_for_project(
            token, supabase_url, supabase_anon_key
        )

    for _, supabase_url, supabase_anon_key in configs:
        identity = await _verify_supabase_token_for_project(
            token, supabase_url, supabase_anon_key
        )
        if identity:
            return identity
    return None


def _allow_legacy_auth() -> bool:
    return os.getenv("ALLOW_LEGACY_USER_ID_AUTH", "true").lower() not in {
        "0",
        "false",
        "no",
    }


def _legacy_identity(token: str) -> str | None:
    if not _allow_legacy_auth():
        return None
    if token.startswith("user-") or token.startswith("polly-") or "@" in token:
        return token
    return None


@auth.authenticate
async def authenticate(
    authorization: str | None, headers: dict
) -> Auth.types.MinimalUserDict:
    """Validate requests and extract user identity.

    Real LangGraph Studio users are supplied by LangGraph itself and handled in
    authorization callbacks with is_studio_user(). Public API callers must
    present a Supabase token, a signed guest token, or a temporary legacy raw ID.
    """
    token = _extract_bearer_token(authorization)
    if not token:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Authentication required"
        )

    client_ip = _get_client_ip(headers)
    guest_identity = _verify_guest_token(token)
    if guest_identity:
        return {
            "identity": guest_identity,
            "is_authenticated": True,
            "auth_type": "guest",
            "client_ip": client_ip,
        }

    selected_region = _get_header(headers, "x-supabase-region")
    if selected_region not in AUTH_REGIONS:
        selected_region = None
    supabase_identity = await _verify_supabase_token(token, selected_region)
    if supabase_identity:
        return {
            "identity": supabase_identity,
            "is_authenticated": True,
            "auth_type": "supabase",
            "client_ip": client_ip,
        }

    legacy_identity = _legacy_identity(token)
    if legacy_identity:
        return {
            "identity": legacy_identity,
            "is_authenticated": True,
            "auth_type": "legacy",
            "client_ip": client_ip,
        }

    raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")


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

    user_id = _get_user_field(ctx.user, "identity")
    if not isinstance(user_id, str) or not user_id:
        raise Auth.exceptions.HTTPException(401, "Invalid authenticated user")
    metadata = value.setdefault("metadata", {})
    metadata["user_id"] = user_id

    return {"user_id": user_id}


@auth.on.threads.update
async def update_owner_metadata(ctx: Auth.types.AuthContext, value: dict):
    """Allow users to update metadata only on their own threads."""
    if is_studio_user(ctx.user):
        return {}

    user_id = _get_user_field(ctx.user, "identity")
    if not isinstance(user_id, str) or not user_id:
        raise Auth.exceptions.HTTPException(401, "Invalid authenticated user")
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
    validate_inputs(
        value["kwargs"].get("input"), value["kwargs"].get("command")
    )
    validate_config(value["kwargs"].get("config") or value.get("config"))
    if is_studio_user(ctx.user):
        return {}

    user_id = _get_user_field(ctx.user, "identity")
    if not isinstance(user_id, str) or not user_id:
        raise Auth.exceptions.HTTPException(401, "Invalid authenticated user")
    client_ip = _get_user_field(ctx.user, "client_ip")
    _check_rate_limit_for_ip(client_ip if isinstance(client_ip, str) else "unknown")

    metadata["user_id"] = user_id
    return {"user_id": user_id}


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

    user_id = _get_user_field(ctx.user, "identity")
    if not isinstance(user_id, str) or not user_id:
        raise Auth.exceptions.HTTPException(401, "Invalid authenticated user")
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


def validate_config(config: dict | None):
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

    configurable.pop("model", None)
    configurable.pop("model_provider", None)


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
