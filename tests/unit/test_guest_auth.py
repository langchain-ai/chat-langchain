"""Guest pass issuance and the local check the agent uses."""

import managed_deepagents._validated_token as validated_token
from managed_deepagents.runtime import build_managed_http_app
from starlette.testclient import TestClient

from guest_auth import read_guest_token, sign_guest_token
from identity import identity

_SECRET = "test-guest-secret"
_PROVIDER = {
    "id": "guest",
    "issuer": "mda:guest",
    "claims": {"user": "sub"},
    "introspect": {"url": "http://127.0.0.1/unused"},
}


def test_guest_route_issues_a_pass_the_agent_accepts(monkeypatch):
    monkeypatch.setenv("GUEST_AUTH_SECRET", _SECRET)
    app = build_managed_http_app(identity=identity, connectors=[])
    client = TestClient(app)

    issued = client.post("/identity/guest")
    assert issued.status_code == 200
    body = issued.json()
    claims = read_guest_token(_SECRET, body["token"])
    assert claims is not None
    assert claims["iss"] == "mda:guest"
    assert body["guestId"] == claims["sub"]
    assert claims["sub"].startswith("guest:")

    checked = validated_token._verify_provider_claims(_PROVIDER, body["token"], {})
    assert checked["sub"] == claims["sub"]

    introspected = client.get(
        "/identity/guest/introspect",
        headers={"authorization": f"Bearer {body['token']}"},
    )
    assert introspected.status_code == 200
    assert introspected.json()["sub"] == claims["sub"]


def test_guest_check_rejects_a_bad_pass(monkeypatch):
    monkeypatch.setenv("GUEST_AUTH_SECRET", _SECRET)
    token = sign_guest_token(_SECRET, "guest:someone")
    tampered = token[:-2] + ("aa" if not token.endswith("aa") else "bb")

    try:
        validated_token._verify_provider_claims(_PROVIDER, tampered, {})
    except Exception as error:
        assert getattr(error, "status", None) == 401
    else:
        raise AssertionError("bad guest pass was accepted")


def test_guest_route_is_unavailable_without_a_secret(monkeypatch):
    monkeypatch.delenv("GUEST_AUTH_SECRET", raising=False)
    app = build_managed_http_app(identity=identity, connectors=[])
    response = TestClient(app).post("/identity/guest")
    assert response.status_code == 503
