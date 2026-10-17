"""Guest passes for callers who are not signed in.

The docs chat asks this agent for a pass at ``POST /identity/guest``. MDA 0.9
does not serve that route, and ``mda`` rewrites ``_mda_http.py`` on every
build, so the route is attached when that generated module imports identity.
The agent checks the pass locally with ``GUEST_AUTH_SECRET`` instead of
calling back out to itself.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from importlib import import_module
from typing import Any

GUEST_TOKEN_ISSUER = "mda:guest"
_GUEST_TTL_SECONDS = 24 * 60 * 60
_INSTALLED = False


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _sign(secret: str, header: str, body: str) -> str:
    return _b64url(
        hmac.new(secret.encode("utf-8"), f"{header}.{body}".encode("ascii"), hashlib.sha256).digest()
    )


def sign_guest_token(secret: str, subject: str, now: float | None = None) -> str:
    """Return an HS256 guest JWT whose ``sub`` is ``subject``."""
    issued = int(time.time() if now is None else now)
    header = _b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64url(
        json.dumps(
            {
                "iss": GUEST_TOKEN_ISSUER,
                "sub": subject,
                "iat": issued,
                "exp": issued + _GUEST_TTL_SECONDS,
            },
            separators=(",", ":"),
        ).encode()
    )
    return f"{header}.{body}.{_sign(secret, header, body)}"


def read_guest_token(secret: str, token: str, now: float | None = None) -> dict[str, Any] | None:
    """Return the guest claims, or ``None`` when the token is not valid."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    header, body, signature = parts
    expected = _sign(secret, header, body)
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        padded = body + "=" * ((4 - len(body) % 4) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(claims, dict):
        return None
    subject = claims.get("sub")
    expires = claims.get("exp")
    current = time.time() if now is None else now
    if claims.get("iss") != GUEST_TOKEN_ISSUER:
        return None
    if not isinstance(subject, str) or not subject.startswith("guest:"):
        return None
    if not isinstance(expires, int) or expires <= current:
        return None
    return claims


def _secret() -> str:
    return os.environ.get("GUEST_AUTH_SECRET") or ""


def _json(payload: dict[str, Any], status: int) -> Any:
    responses = import_module("starlette.responses")
    return responses.JSONResponse(payload, status_code=status)


async def _issue_guest(_request: Any) -> Any:
    secret = _secret()
    if not secret:
        return _json({"detail": "Guest authentication is unavailable"}, 503)
    subject = f"guest:{uuid.uuid4()}"
    return _json({"token": sign_guest_token(secret, subject), "guestId": subject}, 200)


async def _introspect_guest(request: Any) -> Any:
    secret = _secret()
    if not secret:
        return _json({"detail": "Guest authentication is unavailable"}, 503)
    authorization = request.headers.get("authorization") or ""
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    claims = read_guest_token(secret, token) if token else None
    if claims is None:
        return _json({"detail": "Unauthorized"}, 401)
    return _json({"iss": claims["iss"], "sub": claims["sub"]}, 200)


def _guest_routes() -> list[Any]:
    routing = import_module("starlette.routing")
    return [
        routing.Route("/identity/guest", _issue_guest, methods=["POST"]),
        routing.Route("/identity/guest/introspect", _introspect_guest, methods=["GET"]),
    ]


def _install_route_hook() -> None:
    applications = import_module("starlette.applications")
    original = applications.Starlette.__init__
    if getattr(original, "_chat_langchain_guest_hook", False):
        return

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        paths = {getattr(route, "path", None) for route in self.router.routes}
        if "/identity/guest" in paths:
            return
        self.router.routes[0:0] = _guest_routes()

    init._chat_langchain_guest_hook = True  # type: ignore[attr-defined]
    applications.Starlette.__init__ = init  # type: ignore[method-assign]


def _install_local_check() -> None:
    tokens = import_module("managed_deepagents._validated_token")
    original = tokens._verify_provider_claims
    if getattr(original, "_chat_langchain_guest_check", False):
        return

    def verify(provider: dict[str, Any], token: str, headers: Any) -> Any:
        if provider.get("id") != "guest":
            return original(provider, token, headers)
        claims = read_guest_token(_secret(), token) if _secret() else None
        if claims is None:
            error = import_module("managed_deepagents._identity_auth").ManagedAuthError
            raise error(401, "token introspection failed")
        return claims

    verify._chat_langchain_guest_check = True  # type: ignore[attr-defined]
    tokens._verify_provider_claims = verify


def install_guest_auth() -> None:
    """Attach the public guest route and the local pass check once."""
    global _INSTALLED
    if _INSTALLED:
        return
    _install_route_hook()
    _install_local_check()
    _INSTALLED = True
