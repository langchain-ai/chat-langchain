import hashlib

import pytest


@pytest.fixture
def identity_module(monkeypatch):
    monkeypatch.setenv("IDENTITY_ACTOR_SALT", "synthetic-test-salt")
    import identity

    return identity


def test_normalize_actor_id_prefers_subject_and_hashes_email(identity_module):
    email_claims = {"email": "  user@example.test "}
    first = identity_module._normalize_actor_id(email_claims)
    second = identity_module._normalize_actor_id(email_claims)

    assert first == second
    assert first.startswith("user-")
    assert "@" not in first
    expected = hashlib.sha256(b"synthetic-test-salt:user@example.test").hexdigest()[:32]
    assert first == f"user-{expected}"

    assert (
        identity_module._normalize_actor_id(
            {
                "sub": "550e8400-e29b-41d4-a716-446655440000",
                "email": "user@example.test",
            }
        )
        == "user-550e8400-e29b-41d4-a716-446655440000"
    )


def test_supabase_providers_use_subject_claim(identity_module, monkeypatch):
    for region, (url_env, key_env) in identity_module._REGION_ENV.items():
        monkeypatch.setenv(url_env, f"https://{region}.supabase.test")
        monkeypatch.setenv(key_env, "synthetic-anon-key")

    entries = identity_module._providers()

    assert [entry["claims"]["actor"] for entry in entries[:-1]] == ["sub"] * 4
    assert entries[-1] == {
        "id": "guest",
        "guest": {"issue": True, "ttl": "24h", "actor_prefix": "guest:"},
        "claims": {"actor": "sub"},
    }
