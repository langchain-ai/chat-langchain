"""Tests for the deployed MDA identity contract."""

from identity import _supabase_auth, identity


def test_identity_uses_multi_region_supabase_ids():
    http = identity.config["ingress"]["http"]
    assert http["mode"] == "validated_token"
    assert len(http["providers"]) == 2

    provider = http["providers"][0]
    guest = http["providers"][1]
    assert guest["issuer"] == "mda:guest"
    assert guest["claims"] == {"user": "sub"}
    assert guest["introspect"]["url"] == "${GUEST_INTROSPECT_URL}"

    assert provider["claims"] == {"user": "id", "email": "email"}
    assert provider["introspect"]["region_header"] == "x-supabase-region"
    assert provider["introspect"]["regions"] == {
        "us": {
            "url": "${SUPABASE_URL}/auth/v1/user",
            "headers": {"apikey": "${SUPABASE_ANON_KEY}"},
        },
        "eu": {
            "url": "${SUPABASE_EU_URL}/auth/v1/user",
            "headers": {"apikey": "${SUPABASE_EU_ANON_KEY}"},
        },
        "apac": {
            "url": "${SUPABASE_APAC_URL}/auth/v1/user",
            "headers": {"apikey": "${SUPABASE_APAC_ANON_KEY}"},
        },
        "aws": {
            "url": "${SUPABASE_AWS_URL}/auth/v1/user",
            "headers": {"apikey": "${SUPABASE_AWS_ANON_KEY}"},
        },
    }


def test_identity_resolves_runtime_urls_for_principal_issuers(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://us.example.com/")
    monkeypatch.setenv("SUPABASE_EU_URL", "https://eu.example.com/auth/v1")

    provider = _supabase_auth()

    assert provider["introspect"]["regions"]["us"]["url"] == (
        "https://us.example.com/auth/v1/user"
    )
    assert provider["introspect"]["regions"]["eu"]["url"] == (
        "https://eu.example.com/auth/v1/user"
    )
    assert provider["issuers"][:2] == [
        "https://us.example.com/auth/v1",
        "https://eu.example.com/auth/v1",
    ]
