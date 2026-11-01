"""Validate HTTP input and sanitize human messages before model calls."""

from __future__ import annotations

import base64
import binascii
import io
import json
import re
import warnings
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import convert_to_messages
from langgraph.runtime import Runtime
from PIL import Image
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


class IngressGuardsMiddleware(AgentMiddleware):
    """Cap user text and sanitize images at agent ingress."""

    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Sanitize all human messages, preserving their IDs for state updates."""
        updates = []
        for message in state.get("messages", []):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                if capped is not message.content:
                    updates.append(message.model_copy(update={"content": capped}))
        return {"messages": updates} if updates else None

    def _truncate_content(self, content: Any) -> Any:
        """Cap user text and replace unsafe or invalid images with text."""
        if isinstance(content, str):
            return (
                content[:MAX_MESSAGE_CHARS]
                if len(content) > MAX_MESSAGE_CHARS
                else content
            )

        if not isinstance(content, list):
            return content

        remaining = MAX_MESSAGE_CHARS
        changed = False
        truncated: list[Any] = []
        for block in content:
            if isinstance(block, str):
                text = block[:remaining]
                changed = changed or len(text) != len(block)
                truncated.append(text)
                remaining -= len(text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = block["text"][:remaining]
                changed = changed or len(text) != len(block["text"])
                truncated.append({**block, "text": text})
                remaining -= len(text)
            elif isinstance(block, dict) and block.get("type") in (
                "image_url",
                "image",
                "input_image",
            ):
                if block.get("type") == "image_url" and _valid_image(
                    block.get("image_url")
                ):
                    truncated.append(block)
                else:
                    changed = True
                    truncated.append(
                        {"type": "text", "text": "[unsupported image removed]"}
                    )
            else:
                truncated.append(block)
        return truncated if changed else content


def _valid_image(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("url"), str):
        return False
    if value.get("detail", "auto") not in ("auto", "low", "high"):
        return False
    match = re.fullmatch(r"data:image/(png|jpeg|gif|webp);base64,(.+)", value["url"])
    if match is None:
        return False
    try:
        data = base64.b64decode(match[2], validate=True)
        if match[1] == "png" and not data.endswith(
            b"\x00\x00\x00\x00IEND\xaeB\x60\x82"
        ):
            return False
        if match[1] == "gif" and not data.endswith(b";"):
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format != match[1].upper():
                    return False
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                for frame in range(getattr(image, "n_frames", 1)):
                    image.seek(frame)
                    image.load()
    except (
        ValueError,
        binascii.Error,
        OSError,
        SyntaxError,
        EOFError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ):
        return False
    return True


def _validate_input(payload: Any) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Run input must be an object")
    config = payload.get("config")
    if isinstance(config, dict) and "recursion_limit" in config:
        limit = config["recursion_limit"]
        if type(limit) is not int or limit < 1:
            raise ValueError("recursion_limit must be a positive integer")
    inputs = payload.get("input", payload.get("values"))
    if inputs is None:
        return
    if not isinstance(inputs, dict):
        raise ValueError("Input must be an object")
    if "messages" not in inputs:
        return
    messages = inputs["messages"]
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")
    for entry in messages:
        if not isinstance(entry, dict) or not isinstance(
            entry.get("content"), (str, list)
        ):
            raise ValueError(
                "Each message must be an object with string or list content"
            )
        content = entry["content"]
        if isinstance(content, list) and any(
            not isinstance(b, (str, dict)) for b in content
        ):
            raise ValueError("Message content blocks must be strings or objects")
        if isinstance(content, list) and any(
            isinstance(b, dict)
            and b.get("type") == "text"
            and not isinstance(b.get("text"), str)
            for b in content
        ):
            raise ValueError("Text blocks must contain string text")
        try:
            message = convert_to_messages([entry])[0]
        except (ValueError, TypeError, NotImplementedError, KeyError) as error:
            raise ValueError("Invalid message role or content") from error
        if message.type == "human":
            entry["content"] = IngressGuardsMiddleware()._truncate_content(content)


class IngressInputMiddleware:
    """Validate and sanitize managed HTTP writes before graph reduction."""

    def __init__(self, app: ASGIApp) -> None:
        """Wrap the managed HTTP application."""
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Reject malformed messages and replay sanitized input to the runtime."""
        path = scope.get("path", "").rstrip("/")
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or not re.fullmatch(
                r"/(?:threads/[^/]+/)?runs(?:/(?:stream|wait|batch))?|/threads/[^/]+/state",
                path,
            )
        ):
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive)
        try:
            payload = await request.json()
            if path.endswith("/runs/batch") and isinstance(payload, list):
                for item in payload:
                    _validate_input(item)
            else:
                _validate_input(payload)
        except (ValueError, UnicodeDecodeError) as error:
            await JSONResponse({"detail": str(error)}, status_code=422)(
                scope, receive, send
            )
            return
        body = json.dumps(payload).encode()
        scope = {
            **scope,
            "headers": [
                (key, value)
                for key, value in scope.get("headers", [])
                if key != b"content-length"
            ]
            + [(b"content-length", str(len(body)).encode())],
        }

        async def sanitized_receive() -> Message:
            nonlocal body
            if body:
                data, body = body, b""
                return {"type": "http.request", "body": data, "more_body": False}
            return await receive()

        await self.app(scope, sanitized_receive, send)


__all__ = ["IngressGuardsMiddleware", "IngressInputMiddleware", "MAX_MESSAGE_CHARS"]
