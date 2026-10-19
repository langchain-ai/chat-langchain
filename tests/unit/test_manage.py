from unittest.mock import Mock

import pytest
import requests

from scripts import manage


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        '# Deployment settings\n\nPYLON_API_KEY="secret=test-key"\n'
        "PYLON_KB_ID=kb-test\n"
    )
    monkeypatch.setattr(manage.sys, "argv", ["manage.py", "deploy", "--name", "test"])
    build = Mock(return_value=tmp_path / "langgraph.json")
    run = Mock()
    monkeypatch.setattr(manage, "build", build)
    monkeypatch.setattr(manage.subprocess, "run", run)
    return build, run


@pytest.mark.parametrize("status", [401, 403, 404, 500, 302])
def test_deploy_aborts_on_failed_preflight(deployment, monkeypatch, status, capsys):
    build, run = deployment
    monkeypatch.setattr(
        manage.requests, "get", Mock(return_value=Mock(status_code=status))
    )

    with pytest.raises(SystemExit) as error:
        manage.main()

    assert f"HTTP {status}" in str(error.value)
    if status in (401, 403):
        assert "rotate PYLON_API_KEY" in str(error.value)
    assert "secret=test-key" not in str(error.value)
    assert "secret=test-key" not in str(capsys.readouterr())
    build.assert_not_called()
    run.assert_not_called()


def test_deploy_proceeds_after_successful_preflight(deployment, monkeypatch):
    build, run = deployment
    get = Mock(return_value=Mock(status_code=200))
    monkeypatch.setattr(manage.requests, "get", get)

    manage.main()

    get.assert_called_once_with(
        "https://api.usepylon.com/knowledge-bases/kb-test/articles",
        headers={"Authorization": "Bearer secret=test-key"},
        timeout=10,
        allow_redirects=False,
    )
    run.assert_called_once_with(
        [
            manage.sys.executable,
            "-m",
            "langgraph_cli",
            "deploy",
            "--config",
            str(build.return_value),
            "--name",
            "test",
        ],
        check=True,
        cwd=manage.ROOT,
    )


@pytest.mark.parametrize("contents", [None, "PYLON_API_KEY=key\n", "PYLON_KB_ID=kb\n"])
def test_deploy_requires_project_environment(deployment, monkeypatch, contents):
    build, run = deployment
    env = manage.ROOT / ".env"
    if contents is None:
        env.unlink()
    else:
        env.write_text(contents)
    get = Mock()
    monkeypatch.setattr(manage.requests, "get", get)

    with pytest.raises(SystemExit, match="project .env"):
        manage.main()

    get.assert_not_called()
    build.assert_not_called()
    run.assert_not_called()


def test_deploy_sanitizes_request_failure(deployment, monkeypatch):
    build, run = deployment
    monkeypatch.setattr(
        manage.requests, "get", Mock(side_effect=requests.Timeout("secret=test-key"))
    )

    with pytest.raises(SystemExit) as error:
        manage.main()

    assert str(error.value) == "Deployment aborted: Pylon preflight request failed."
    assert error.value.__suppress_context__
    build.assert_not_called()
    run.assert_not_called()


def test_build_does_not_require_pylon(deployment, monkeypatch):
    build, run = deployment
    monkeypatch.setattr(manage.sys, "argv", ["manage.py", "build"])
    get = Mock()
    monkeypatch.setattr(manage.requests, "get", get)

    manage.main()

    build.assert_called_once()
    get.assert_not_called()
    run.assert_not_called()
