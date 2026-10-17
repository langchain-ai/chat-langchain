"""Managed Deep Agent identity contract for Chat LangChain.

Signed-in callers present a Supabase access token. Guests present a pass
from ``POST /identity/guest`` (``iss`` ``mda:guest``). The agent checks that
pass with ``GUEST_AUTH_SECRET``.
"""

from __future__ import annotations

import os

from managed_deepagents import auth, define_identity

from guest_auth import install_guest_auth

# Region label (sent by the frontend as ``x-supabase-region``) -> the env vars
# that carry that region's Supabase project URL and anon key. Matches the
# regions the previous ``auth.py`` supported.
_REGION_ENV: dict[str, tuple[str, str]] = {
    "us": ("SUPABASE_URL", "SUPABASE_ANON_KEY"),
    "eu": ("SUPABASE_EU_URL", "SUPABASE_EU_ANON_KEY"),
    "apac": ("SUPABASE_APAC_URL", "SUPABASE_APAC_ANON_KEY"),
    "aws": ("SUPABASE_AWS_URL", "SUPABASE_AWS_ANON_KEY"),
}


def _supabase_user_url(url: str) -> str:
    base = url.rstrip("/")
    if base.endswith("/auth/v1/user"):
        return base
    if base.endswith("/auth/v1"):
        return f"{base}/user"
    return f"{base}/auth/v1/user"


def _supabase_auth() -> dict:
    regions = {
        region: {
            "url": _supabase_user_url(os.environ.get(url_env) or f"${{{url_env}}}"),
            "anon_key": f"${{{key_env}}}",
        }
        for region, (url_env, key_env) in _REGION_ENV.items()
    }
    provider = auth.supabase(
        introspect=True,
        regions=regions,
        region_header="x-supabase-region",
    )
    # Supabase anonymous users have no email. The stable `id` returned by
    # `/auth/v1/user` identifies both anonymous and signed-in callers.
    provider["claims"] = {"user": "id", "email": "email"}
    return provider


def _guest_auth() -> dict:
    return {
        "id": "guest",
        "issuer": "mda:guest",
        "claims": {"user": "sub"},
        # MDA requires an introspect URL. The auth path checks the signature
        # locally; this route is the same check over HTTP.
        "introspect": {"url": "${MDA_PUBLIC_API_URL}/identity/guest/introspect"},
    }


identity = define_identity(
    auth=[_supabase_auth(), _guest_auth()],
)
install_guest_auth()
