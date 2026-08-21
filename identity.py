"""Managed Deep Agent identity contract for Chat LangChain.

Auto-discovered by the compiler (never imported by ``agent.py``). Replaces the
hand-rolled ``src/api/auth.py`` handler:

- ``validated_token`` ingress: the browser calls the deployment directly with its
  own Supabase access token (or an MDA-issued guest token); MDA verifies it.
- Multi-region Supabase: one provider per ``SUPABASE_*_URL`` via
  ``providers.supabase(url=..., introspect=True)``. Custom auth domains set
  ``discovery_url`` so MDA resolves the real JWT ``iss`` for routing; verification
  uses ``/auth/v1/user`` introspection (required while some regions still mint
  legacy HS256 tokens with an empty JWKS).
- ``threads: "actor"`` replaces the ``@auth.on.threads`` owner tagging. The actor
  is always a pseudonymous id -- the Supabase uid returned by introspection, or
  the guest actor id -- never the email claim: the actor is copied verbatim into
  the thread ``owner`` and into traced run metadata
  (``owner``/``langgraph_auth_user_id``/``user_id``), so raw email addresses and
  personal names must never reach it.
"""

from __future__ import annotations

import os

from managed_deepagents import define_identity, providers

# Region label (sent by the frontend as ``x-supabase-region``) -> the env vars
# that carry that region's Supabase project URL and anon key. Matches the
# regions the previous ``auth.py`` supported.
_REGION_ENV: dict[str, tuple[str, str]] = {
    "us": ("SUPABASE_URL", "SUPABASE_ANON_KEY"),
    "eu": ("SUPABASE_EU_URL", "SUPABASE_EU_ANON_KEY"),
    "apac": ("SUPABASE_APAC_URL", "SUPABASE_APAC_ANON_KEY"),
    "aws": ("SUPABASE_AWS_URL", "SUPABASE_AWS_ANON_KEY"),
}

# ``providers.supabase(...)`` defaults the actor to the ``email`` claim, which
# leaks the address into thread ownership and traced run metadata. ``/auth/v1/user``
# returns the opaque Supabase uid as ``id`` (introspection responses carry no
# ``sub``), so map the actor to that instead.
_ACTOR_CLAIM = "id"


def _providers() -> list[dict]:
    entries: list[dict] = []
    for region, (url_env, key_env) in _REGION_ENV.items():
        base = os.environ.get(url_env)
        if not base:
            continue
        # Introspect: US/EU JWKS are empty (legacy HS256). discovery_url still
        # resolves the real *.supabase.co iss for multi-provider routing.
        provider = providers.supabase(url=base.rstrip("/"), introspect=True)
        provider["id"] = f"supabase-{region}"
        provider["introspect"]["headers"] = {"apikey": "${" + key_env + "}"}
        provider["claims"] = {"actor": _ACTOR_CLAIM}
        entries.append(provider)
    # Anonymous visitors: MDA issues + verifies signed guest tokens itself
    # (POST /identity/guest), replacing the frontend guest-token route.
    entries.append(providers.guest(ttl="24h", actor_prefix="guest:"))
    configured = [
        e.get("discovery_url") or e.get("issuer")
        for e in entries
        if e.get("discovery_url") or e.get("issuer")
    ]
    print(f"[chat-langchain identity] supabase discovery bases: {configured}", flush=True)
    return entries


identity = define_identity(
    ingress={"http": {"mode": "validated_token", "providers": _providers()}},
    tenancy="single",
    scoping={"threads": "actor", "memory": "none", "credentials": "agent"},
)
