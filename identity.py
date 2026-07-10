"""Managed Deep Agent identity contract for Chat LangChain.

Auto-discovered by the compiler (never imported by ``agent.py``). Replaces the
hand-rolled ``src/api/auth.py`` handler:

- ``validated_token`` ingress: the browser calls the deployment directly with its
  own Supabase access token (or an MDA-issued guest token); MDA verifies it.
- Multi-region Supabase: one JWKS provider per project so MDA routes by token
  ``iss`` (required when guest is also configured). Non-standard hosts fall back
  to issuer + introspection.
- ``threads: "actor"`` replaces the ``@auth.on.threads`` owner tagging (owner =
  the user's email, or the guest actor id).

Supabase region URLs are resolved from the same environment variables the old
handler used (``SUPABASE_URL`` / ``SUPABASE_EU_URL`` / ...). They are read when
``identity.py`` is imported so each region's project ref/issuer is known up
front; per-region anon keys for the introspect fallback stay ``${ENV}``
placeholders expanded at request time.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from managed_deepagents import define_identity, providers
from managed_deepagents._identity_auth import ManagedAuthError
from managed_deepagents import _validated_token as _vt


def _install_issuer_mismatch_detail() -> None:
    """Include the token ``iss`` and configured issuers in the 401 detail.

    The stock MDA error is only ``no provider matches token issuer``, which is
    not enough to debug frontend-vs-secret project skew after deploy.
    """
    original = _vt.select_provider

    def select_provider_with_detail(providers, token):  # noqa: ANN001
        try:
            return original(providers, token)
        except ManagedAuthError as err:
            if err.status != 401 or "no provider matches token issuer" not in err.detail:
                raise
            issuer = _vt._try_unverified_issuer(token)
            configured = [p.get("issuer") for p in providers if p.get("issuer")]
            raise ManagedAuthError(
                401,
                f"no provider matches token issuer {issuer!r}; configured={configured!r}",
            ) from err

    _vt.select_provider = select_provider_with_detail


_install_issuer_mismatch_detail()

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


def _supabase_provider(region: str, region_config: dict[str, str]) -> dict:
    """One provider per region so MDA can route JWTs by token ``iss``.

    Multi-provider selection matches ``iss`` exactly. A single introspect
    provider with no ``issuer`` (plus guest) therefore rejects every real
    Supabase access token with ``no provider matches token issuer``. Emit one
    JWKS provider per project ref instead; fall back to issuer+introspect only
    for non-standard project URLs.
    """
    project_ref = region_config.get("project_ref")
    if project_ref:
        provider = providers.supabase(project_ref=project_ref)
        provider["id"] = f"supabase-{region}"
        return provider

    # Custom / non-supabase.co host: keep introspection, but still set issuer
    # from the configured URL so multi-provider routing can find this entry.
    url = region_config["url"]  # .../auth/v1/user
    issuer = url[: -len("/user")] if url.endswith("/user") else url.rstrip("/")
    return {
        "id": f"supabase-{region}",
        "issuer": issuer,
        "introspect": {
            "url": url,
            "headers": {"apikey": region_config["anon_key"]},
        },
        "claims": {"actor": "email"},
    }


def _providers() -> list[dict]:
    entries: list[dict] = []
    # One entry per region/project so each token ``iss`` selects a verifier.
    # Do not collapse multi-region into a single issuer-less introspect provider.
    for region, region_config in _supabase_regions().items():
        entries.append(_supabase_provider(region, region_config))
    # Anonymous visitors: MDA issues + verifies signed guest tokens itself
    # (POST /identity/guest), replacing the frontend guest-token route.
    entries.append(providers.guest(ttl="24h", actor_prefix="guest:"))
    configured = [e.get("issuer") for e in entries if e.get("issuer")]
    # Surfaces in LangSmith deploy logs so we can compare to the browser token iss.
    print(f"[chat-langchain identity] configured token issuers: {configured}", flush=True)
    return entries


identity = define_identity(
    ingress={"http": {"mode": "validated_token", "providers": _providers()}},
    tenancy="single",
    scoping={"threads": "actor", "memory": "none", "credentials": "agent"},
)
