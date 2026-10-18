"""Tests for the deployed MDA identity contract."""

import base64
import json

from managed_deepagents import _validated_token

from identity import _REGION_ENV, _guest_auth, _supabase_auths, identity


def _token_with_unverified_issuer(issuer: str) -> str:
    def encode(value: dict) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()

    signature = base64.urlsafe_b64encode(b"test-signature").rstrip(b"=").decode()
    return f"{encode({'alg': 'HS256', 'typ': 'JWT'})}.{encode({'iss': issuer})}.{signature}"


def test_identity_includes_guest_auth():
    http = identity.config["ingress"]["http"]
    assert http["mode"] == "validated_token"

    guest = http["providers"][-1]
    assert guest["issuer"] == "mda:guest"
    assert guest["claims"] == {"user": "sub"}
    assert (
        guest["introspect"]["url"] == "${MDA_PUBLIC_API_URL}/identity/guest/introspect"
    )


def test_supabase_auths_skip_unconfigured_regions(monkeypatch):
    for url_env, _ in _REGION_ENV.values():
        monkeypatch.delenv(url_env, raising=False)
    monkeypatch.setenv("SUPABASE_URL", "https://us.example.com/")
    monkeypatch.setenv("SUPABASE_EU_URL", "https://eu.example.com")

    providers = _supabase_auths()

    assert len(providers) == 2
    assert [provider["id"] for provider in providers] == ["supabase", "supabase"]
    assert [provider["discovery_url"] for provider in providers] == [
        "https://us.example.com/auth/v1",
        "https://eu.example.com/auth/v1",
    ]
    assert [provider["introspect"]["headers"] for provider in providers] == [
        {"apikey": "${SUPABASE_ANON_KEY}"},
        {"apikey": "${SUPABASE_EU_ANON_KEY}"},
    ]
    assert all("issuer" not in provider for provider in providers)
    assert all(
        provider["claims"] == {"user": "id", "email": "email"} for provider in providers
    )


def test_custom_domain_discovers_real_token_issuer(monkeypatch):
    custom_domain = "https://auth.example.com"
    token_issuer = "https://project-ref.supabase.co/auth/v1"
    for url_env, _ in _REGION_ENV.values():
        monkeypatch.delenv(url_env, raising=False)
    monkeypatch.setenv("SUPABASE_URL", custom_domain)
    monkeypatch.setenv("SUPABASE_ANON_KEY", "test-anon-key")

    provider = _supabase_auths()[0]
    assert provider["discovery_url"] == f"{custom_domain}/auth/v1"

    def get_json(url: str, headers: dict) -> dict:
        if url.endswith("/.well-known/openid-configuration"):
            return {
                "issuer": token_issuer,
                "jwks_uri": f"{token_issuer}/.well-known/jwks.json",
            }
        assert url == f"{custom_domain}/auth/v1/user"
        assert headers["apikey"] == "test-anon-key"
        return {
            "id": "user-123",
            "email": "person@example.com",
            "iss": token_issuer,
        }

    monkeypatch.setattr(_validated_token, "_http_get_json", get_json)
    user = _validated_token.resolve_validated_token_user(
        {},
        {"mode": "validated_token", "providers": [provider, _guest_auth()]},
        {},
        f"Bearer {_token_with_unverified_issuer(token_issuer)}",
    )

    assert user["identity"] == "user-123"
    assert user["mda_identity_provider"] == "supabase"
    assert user["mda_identity_issuer"] == token_issuer
