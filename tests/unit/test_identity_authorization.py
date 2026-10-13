"""Tests for the compiled deployment's authorization policy."""

import asyncio
import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph_sdk import Auth
from managed_deepagents._validated_token import claims_to_user, issue_guest_token

from identity import identity

OPERATIONS = ["create", "update", "delete", "search", "read"]
USERS = [
    {"identity": "guest:abc", "mda_actor_id": "guest:abc", "permissions": []},
    claims_to_user(
        {"claims": {"actor": "sub", "email": "email"}},
        {"sub": "supabase-user", "email": "user@example.com", "role": "authenticated"},
    ),
]


@pytest.fixture(scope="module")
def deployment_auth():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["mda", "build", str(root)], check=True, capture_output=True)
    config = json.loads((root / ".mda/build/langgraph.json").read_text())
    auth_path, export = config["auth"]["path"].split(":")
    assert config["auth"]["disable_studio_auth"] is False
    return runpy.run_path(str(root / ".mda/build" / auth_path))[export]


def _authorize(auth, user, resource, action, value):
    ctx = SimpleNamespace(user=user, resource=resource, action=action)
    for key in [(resource, action), (resource, "*"), ("*", action), ("*", "*")]:
        if key in auth._handlers:
            handler = auth._handlers[key][-1]
            break
    else:
        assert auth._global_handlers
        handler = auth._global_handlers[-1]
    return asyncio.run(handler(ctx=ctx, value=value))


@pytest.mark.parametrize("user", USERS, ids=["guest", "supabase"])
@pytest.mark.parametrize("action", OPERATIONS)
def test_end_users_cannot_manage_crons(deployment_auth, user, action):
    with pytest.raises(Auth.exceptions.HTTPException) as error:
        _authorize(deployment_auth, user, "crons", action, {})
    assert error.value.status_code == 403


@pytest.mark.parametrize("action", OPERATIONS)
def test_studio_can_manage_crons(deployment_auth, action):
    user = Auth.types.StudioUser("operator", is_authenticated=True)
    assert _authorize(deployment_auth, user, "crons", action, {}) is None


@pytest.mark.parametrize("user", USERS, ids=["guest", "supabase"])
@pytest.mark.parametrize("action", ["create", "update", "delete"])
def test_end_users_cannot_mutate_assistants(deployment_auth, user, action):
    with pytest.raises(Auth.exceptions.HTTPException) as error:
        _authorize(deployment_auth, user, "assistants", action, {})
    assert error.value.status_code == 403


@pytest.mark.parametrize("action", OPERATIONS)
def test_studio_can_manage_assistants(deployment_auth, action):
    user = Auth.types.StudioUser("operator", is_authenticated=True)
    assert _authorize(deployment_auth, user, "assistants", action, {}) is None


@pytest.mark.parametrize("user", USERS, ids=["guest", "supabase"])
@pytest.mark.parametrize("action", ["read", "search"])
def test_end_users_can_read_assistants(deployment_auth, user, action):
    assert _authorize(deployment_auth, user, "assistants", action, {}) is None


@pytest.mark.parametrize("user", USERS, ids=["guest", "supabase"])
@pytest.mark.parametrize("action", [*OPERATIONS, "create_run"])
def test_threads_remain_actor_scoped(deployment_auth, user, action):
    value = {}
    filters = _authorize(deployment_auth, user, "threads", action, value)
    assert identity.config["scoping"]["threads"] == "actor"
    assert filters == {"owner": user["mda_actor_id"]}
    assert value["metadata"]["owner"] == user["mda_actor_id"]
    assert filters["owner"] != "another-actor"


def test_studio_thread_access_remains_unscoped(deployment_auth):
    user = Auth.types.StudioUser("operator", is_authenticated=True)
    value = {}
    assert _authorize(deployment_auth, user, "threads", "read", value) is None
    assert value == {}


@pytest.mark.parametrize("user", USERS, ids=["guest", "supabase"])
def test_unlisted_resources_are_denied_by_default(deployment_auth, user):
    with pytest.raises(Auth.exceptions.HTTPException) as error:
        _authorize(deployment_auth, user, "unlisted", "create", {})
    assert error.value.status_code == 403


def test_token_claims_cannot_grant_studio_access(deployment_auth):
    user = claims_to_user(
        {"claims": {"actor": "sub"}},
        {"sub": "supabase-user", "kind": "StudioUser", "role": "admin"},
    )
    with pytest.raises(Auth.exceptions.HTTPException) as error:
        _authorize(deployment_auth, user, "crons", "create", {})
    assert error.value.status_code == 403


def test_verified_guest_token_cannot_create_scheduled_run(deployment_auth, monkeypatch):
    monkeypatch.setenv(
        "MDA_GUEST_SIGNING_KEY", "test-only-signing-key-at-least-32-bytes"
    )
    provider = next(
        provider
        for provider in identity.config["ingress"]["http"]["providers"]
        if provider.get("guest")
    )
    token = issue_guest_token(provider)
    user = asyncio.run(
        deployment_auth._authenticate_handler(authorization=f"Bearer {token}")
    )
    assert user["mda_actor_id"].startswith("guest:")
    payload = {
        "assistant_id": "docs_agent",
        "schedule": "*/2 * * * *",
        "input": {"messages": [{"role": "user", "content": "Explain LangGraph"}]},
    }
    with pytest.raises(Auth.exceptions.HTTPException) as error:
        _authorize(deployment_auth, user, "crons", "create", payload)
    assert error.value.status_code == 403
