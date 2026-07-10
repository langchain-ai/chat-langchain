"""Managed Deep Agent identity contract for Chat LangChain.

Auto-discovered by the compiler (never imported by ``agent.py``). Replaces the
hand-rolled ``src/api/auth.py`` handler:

- ``validated_token`` ingress: the browser calls the deployment directly with its
  own Supabase access token (or an MDA-issued guest token); MDA verifies it.
- Multi-region Supabase: one JWKS provider per project so MDA routes by token
  ``iss`` (required when guest is also configured). Custom auth domains
  (e.g. ``auth.langchain.com``) are resolved via OIDC discovery so the provider
  ``issuer`` matches the real JWT ``iss`` (``*.supabase.co/auth/v1``), not the
  vanity hostname.
- ``threads: "actor"`` replaces the ``@auth.on.threads`` owner tagging (owner =
  the user's email, or the guest actor id).

Supabase region URLs are resolved from the same environment variables the old
handler used (``SUPABASE_URL`` / ``SUPABASE_EU_URL`` / ...). They are read when
``identity.py`` is imported so each region's issuer/JWKS is known up front.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlparse

from managed_deepagents import define_identity, providers
from managed_deepagents import _validated_token as _vt
from managed_deepagents._identity_auth import ManagedAuthError


def _install_issuer_mismatch_detail() -> None:
    """Include the token ``iss`` and configured issuers in the 401 detail."""
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
        # Reject pooler / non-project hosts (e.g. aws-0-….pooler.supabase.com is
        # not .supabase.co; bare *.supabase.co project refs are fine).
        ref = hostname[: -len(suffix)]
        if ref and "." not in ref:
            return ref
    return None


def _auth_v1_base(base: str) -> str:
    """Normalize a Supabase URL to the ``…/auth/v1`` API root."""
    trimmed = base.rstrip("/")
    if trimmed.endswith("/auth/v1"):
        return trimmed
    return f"{trimmed}/auth/v1"


def _discover_supabase_oidc(base: str) -> dict[str, str] | None:
    """Resolve JWT ``issuer`` + JWKS from OIDC discovery.

    Custom domains like ``https://auth.langchain.com`` still mint tokens whose
    ``iss`` is ``https://<project-ref>.supabase.co/auth/v1``. Discovery is the
    reliable way to learn that issuer (and JWKS) from the configured URL.
    """
    discovery = f"{_auth_v1_base(base)}/.well-known/openid-configuration"
    try:
        with urllib.request.urlopen(discovery, timeout=5) as response:
            data = json.loads(response.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, TypeError):
        return None
    issuer = data.get("issuer") if isinstance(data, dict) else None
    jwks = data.get("jwks_uri") if isinstance(data, dict) else None
    if isinstance(issuer, str) and isinstance(jwks, str):
        return {"issuer": issuer, "jwks": jwks}
    return None


def _supabase_regions() -> dict[str, dict[str, str]]:
    """Build the region map from configured env, mirroring the old handler.

    Only regions whose project URL is set are included, so a guest-only local
    deployment (no Supabase configured) still compiles.
    """
    regions: dict[str, dict[str, str]] = {}
    for region, (url_env, key_env) in _REGION_ENV.items():
        base = os.environ.get(url_env)
        if not base:
            continue
        regions[region] = {
            "base": base.rstrip("/"),
            "anon_key": "${" + key_env + "}",
        }
        project_ref = _supabase_project_ref(base)
        if project_ref:
            regions[region]["project_ref"] = project_ref
    return regions


def _supabase_provider(region: str, region_config: dict[str, str]) -> dict:
    """One provider per region so MDA can route JWTs by token ``iss``."""
    project_ref = region_config.get("project_ref")
    if project_ref:
        provider = providers.supabase(project_ref=project_ref)
        provider["id"] = f"supabase-{region}"
        return provider

    base = region_config["base"]
    discovered = _discover_supabase_oidc(base)
    if discovered:
        return {
            "id": f"supabase-{region}",
            "issuer": discovered["issuer"],
            "jwks": discovered["jwks"],
            "algorithms": ["ES256", "RS256"],
            "claims": {"actor": "email"},
        }

    # Last resort: introspect at the configured host. Issuer is best-effort from
    # the vanity URL and may not match token ``iss`` — prefer fixing discovery.
    auth_base = _auth_v1_base(base)
    return {
        "id": f"supabase-{region}",
        "issuer": auth_base,
        "introspect": {
            "url": f"{auth_base}/user",
            "headers": {"apikey": region_config["anon_key"]},
        },
        "claims": {"actor": "email"},
    }


def _providers() -> list[dict]:
    entries: list[dict] = []
    # One entry per region/project so each token ``iss`` selects a verifier.
    for region, region_config in _supabase_regions().items():
        entries.append(_supabase_provider(region, region_config))
    # Anonymous visitors: MDA issues + verifies signed guest tokens itself
    # (POST /identity/guest), replacing the frontend guest-token route.
    entries.append(providers.guest(ttl="24h", actor_prefix="guest:"))
    configured = [e.get("issuer") for e in entries if e.get("issuer")]
    print(f"[chat-langchain identity] configured token issuers: {configured}", flush=True)
    return entries


identity = define_identity(
    ingress={"http": {"mode": "validated_token", "providers": _providers()}},
    tenancy="single",
    scoping={"threads": "actor", "memory": "none", "credentials": "agent"},
)
