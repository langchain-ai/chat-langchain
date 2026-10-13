"""Tests for the compiled managed auth handler and guest permissions."""

import asyncio
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph_sdk import Auth
from managed_deepagents._validated_token import issue_guest_token

from identity import identity


@pytest.fixture(scope="module")
def auth():
    root = Path(__file__).resolve().parents[2]
    subprocess.run(["mda", "build", str(root)], check=True, capture_output=True)
    return runpy.run_path(str(root / ".mda/build/_mda_auth.py"))["auth"]


@pytest.fixture
def guest_user(auth, monkeypatch):
    monkeypatch.setenv(
        "MDA_GUEST_SIGNING_KEY", "test-guest-signing-key-not-for-production"
    )
    provider = next(
        provider
        for provider in identity.config["ingress"]["http"]["providers"]
        if provider.get("guest")
    )
    token = issue_guest_token(provider)
    user = asyncio.run(auth._authenticate_handler(authorization=f"Bearer {token}"))
    assert user["mda_actor_id"].startswith("guest:")
    return user


@pytest.mark.parametrize(
    "action", ["create", "update", "trigger", "read", "search", "delete"]
)
@pytest.mark.parametrize("caller", ["guest", "signed-in", "missing"])
def test_cron_actions_fail_closed(auth, guest_user, action, caller):
    user = guest_user if caller == "guest" else {"identity": "user@example.com"}
    if caller == "missing":
        user = None
    ctx = SimpleNamespace(user=user, resource="crons", action=action)
    handler = auth._handlers[("crons", "*")][0]

    with pytest.raises(Auth.exceptions.HTTPException) as error:
        asyncio.run(handler(ctx, {"schedule": "* * * * *"}))

    assert error.value.status_code == 403


@pytest.mark.parametrize("action", ["create", "create_run"])
def test_guest_can_create_threads_and_runs(auth, guest_user, action):
    ctx = SimpleNamespace(user=guest_user, resource="threads", action=action)
    value = {"metadata": {}}
    handler = auth._handlers[("threads", "*")][0]

    result = asyncio.run(handler(ctx, value))

    assert result == {"owner": guest_user["mda_actor_id"]}
    assert value["metadata"]["owner"] == guest_user["mda_actor_id"]
