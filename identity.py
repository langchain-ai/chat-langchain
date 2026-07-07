"""Managed Deep Agent identity contract for Chat LangChain.

Auto-discovered by the compiler (never imported by ``agent.py``). Replaces the
hand-rolled ``src/api/auth.py`` handler:

- ``validated_token`` ingress: the browser calls the deployment directly with its
  own Supabase access token (or an MDA-issued guest token); MDA verifies it.
- Multi-region Supabase introspection mirrors the previous ``/auth/v1/user``
  region routing, selecting the endpoint by the ``x-supabase-region`` header and
  sending the region's own anon key.
- ``threads: "actor"`` replaces the ``@auth.on.threads`` owner tagging (owner =
  the user's email, or the guest actor id).

Supabase region URLs are resolved from the same environment variables the old
handler used (``SUPABASE_URL`` / ``SUPABASE_EU_URL`` / ...). They are read at
authoring/compile time because MDA needs the introspection URL as a literal; the
per-region anon keys stay ``${ENV}`` placeholders that the managed runtime
expands at request time, so no secret is baked into the build.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

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


def _supabase_project_ref(base: str) -> str | None:
    """Extract the Supabase project ref from a standard project URL."""
    hostname = urlparse(base).hostname
    suffix = ".supabase.co"
    if hostname and hostname.endswith(suffix):
        return hostname[: -len(suffix)]
    return None


def _supabase_regions() -> dict[str, dict[str, str]]:
    """Build the region map from configured env, mirroring the old handler.

    Only regions whose project URL is set are included, so a guest-only local
    deployment (no Supabase configured) still compiles.
    """
    regions: dict[str, dict[str, str]] = {}
    for region, (url_env, key_env) in _REGION_ENV.items():
        base = os.environ.get(url_env)
        if base:
            project_ref = _supabase_project_ref(base)
            if project_ref:
                regions[region] = {
                    "project_ref": project_ref,
                    "anon_key": "${" + key_env + "}",
                }
                continue
            regions[region] = {
                "url": f"{base.rstrip('/')}/auth/v1/user",
                "anon_key": "${" + key_env + "}",
            }
    return regions


def _providers() -> list[dict]:
    entries: list[dict] = []
    regions = _supabase_regions()
    if regions:
        # The JWT auth gate matches providers by token issuer. For the common
        # single-project case, use the project ref so MDA emits issuer + JWKS.
        if len(regions) == 1:
            region_config = next(iter(regions.values()))
            project_ref = region_config.get("project_ref")
            if project_ref:
                entries.append(providers.supabase(project_ref=project_ref))
            else:
                entries.append(
                    providers.supabase(
                        introspect=True,
                        region_header="x-supabase-region",
                        regions=regions,
                    )
                )
        else:
            entries.append(
                providers.supabase(
                    introspect=True,
                    region_header="x-supabase-region",
                    regions=regions,
                )
            )
    # Anonymous visitors: MDA issues + verifies signed guest tokens itself
    # (POST /identity/guest), replacing the frontend guest-token route.
    entries.append(providers.guest(ttl="24h", actor_prefix="guest:"))
    return entries


identity = define_identity(
    ingress={"http": {"mode": "validated_token", "providers": _providers()}},
    tenancy="single",
    scoping={"threads": "actor", "memory": "none", "credentials": "agent"},
)
