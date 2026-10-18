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


def _supabase_auths() -> list[dict]:
    providers: list[dict] = []
    for region, (url_env, key_env) in _REGION_ENV.items():
        base = (os.environ.get(url_env) or f"${{{url_env}}}").rstrip("/")
        provider = auth.supabase(url=base, introspect=True)
        # Agent Auth includes this ID in its principal key. Keep it stable while
        # the discovered issuer distinguishes the regional providers.
        provider["id"] = "supabase"
        provider["introspect"]["headers"] = {"apikey": f"${{{key_env}}}"}
        # Supabase anonymous users have no email. The stable `id` returned by
        # `/auth/v1/user` identifies both anonymous and signed-in callers.
        provider["claims"] = {"user": "id", "email": "email"}
        providers.append(provider)
    return providers


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
    auth=[*_supabase_auths(), _guest_auth()],
)
install_guest_auth()
