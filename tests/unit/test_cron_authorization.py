import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph_sdk import Auth
from managed_deepagents.runtime import build_managed_http_app
from starlette.testclient import TestClient

from identity import identity
from scripts import manage

ACTIONS = ["create", "update", "search", "read", "delete"]


@pytest.fixture(scope="module")
def compiled_auth():
    config_path = manage.build(manage.ROOT / ".mda" / "test-build")
    return load_auth(config_path)


def load_auth(config_path):
    config = json.loads(config_path.read_text())
    module_path, attribute = config["auth"]["path"].split(":")
    spec = importlib.util.spec_from_file_location(
        "compiled_auth", config_path.parent / module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


def authorize(auth, user, resource, action, value):
    context = SimpleNamespace(user=user, resource=resource, action=action)
    handlers = auth._handlers.get(
        (resource, action), auth._handlers.get((resource, "*"), [])
    )
    for handler in handlers:
        result = asyncio.run(handler(ctx=context, value=value))
        if result is not None:
            return result


def test_guest_token_cannot_create_cron(compiled_auth, monkeypatch):
    monkeypatch.setenv(
        "MDA_GUEST_SIGNING_KEY", "test-guest-signing-key-not-for-production"
    )
    with TestClient(build_managed_http_app(identity=identity)) as client:
        response = client.post("/identity/guest")
    assert response.status_code == 200
    user = asyncio.run(
        compiled_auth._authenticate_handler(
            authorization=f"Bearer {response.json()['token']}"
        )
    )
    assert user["mda_actor_id"].startswith("guest:")

    with pytest.raises(Auth.exceptions.HTTPException) as error:
        authorize(
            compiled_auth,
            user,
            "crons",
            "create",
            {"schedule": "* * * * *", "payload": {"assistant_id": "docs_agent"}},
        )
    assert error.value.status_code == 403


@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("actor", ["guest:visitor", "user@example.com"])
def test_end_users_cannot_manage_crons(compiled_auth, action, actor):
    user = {"identity": actor, "mda_actor_id": actor, "permissions": []}
    with pytest.raises(Auth.exceptions.HTTPException) as error:
        authorize(compiled_auth, user, "crons", action, {})
    assert error.value.status_code == 403


@pytest.mark.parametrize("action", ACTIONS)
def test_studio_can_manage_crons(compiled_auth, action):
    assert (
        authorize(
            compiled_auth,
            Auth.types.StudioUser("developer", is_authenticated=True),
            "crons",
            action,
            {},
        )
        is None
    )


def test_thread_and_store_scoping_are_preserved(compiled_auth):
    user = {"identity": "guest:visitor", "mda_actor_id": "guest:visitor"}
    thread = {}
    assert authorize(compiled_auth, user, "threads", "create", thread) == {
        "owner": "guest:visitor"
    }
    assert thread["metadata"]["owner"] == "guest:visitor"
    assert (
        authorize(compiled_auth, user, "store", "get", {"namespace": ["agent"]}) is None
    )


def test_direct_deployment_uses_maintained_auth():
    auth = load_auth(manage.ROOT / "langgraph.json")
    with pytest.raises(Auth.exceptions.HTTPException) as error:
        authorize(auth, {"identity": "guest:visitor"}, "crons", "create", {})
    assert error.value.status_code == 403


def test_deploy_uses_secured_build_without_regenerating(monkeypatch):
    monkeypatch.setattr(manage, "preflight", lambda: None)
    config_path = manage.ROOT / ".mda" / "test-build" / "langgraph.json"
    original_build = manage.build
    monkeypatch.setattr(manage, "build", lambda: original_build(config_path.parent))
    original_run = manage.subprocess.run
    deployed_configs = []

    def run(command, **kwargs):
        if command[2:4] == ["langgraph_cli", "deploy"]:
            deployed = Path(command[command.index("--config") + 1])
            with pytest.raises(Auth.exceptions.HTTPException) as error:
                authorize(
                    load_auth(deployed),
                    {"identity": "guest:visitor"},
                    "crons",
                    "create",
                    {},
                )
            assert error.value.status_code == 403
            deployed_configs.append(json.loads(deployed.read_text()))
            return
        return original_run(command, **kwargs)

    monkeypatch.setattr(manage.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "manage.py",
            "deploy",
            "--name",
            "test-deployment",
            "--deployment-type",
            "prod",
        ],
    )
    manage.main()
    assert len(deployed_configs) == 1
    assert deployed_configs[0]["auth"]["disable_studio_auth"] is False
    assert deployed_configs[0]["env"] == str(manage.ROOT / ".env")
