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
- ``threads: "actor"`` replaces the ``@auth.on.threads`` owner tagging (owner =
  the user's email, or the guest actor id).
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
        entries.append(provider)
    # Anonymous visitors: MDA issues + verifies signed guest tokens itself
    # (POST /identity/guest), replacing the frontend guest-token route.
    # Guest history is addressable only while the actor id in the guest token
    # stays put: threads are actor-scoped, so a re-issued token with a new
    # subject orphans every thread that guest created. A 24h ttl rotated that
    # identity (and so the visible history) roughly daily. 720h == 30d, spelled
    # in hours because "24h" is the only unit spelling known to be accepted.
    entries.append(providers.guest(ttl="720h", actor_prefix="guest:"))
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
