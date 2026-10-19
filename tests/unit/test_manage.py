"""Tests for source revision stamping in archive builds."""

import json
import subprocess

import pytest
from dotenv import dotenv_values

from scripts import manage

SHA = "a" * 40
FALLBACK_SHA = "b" * 40


@pytest.mark.parametrize("git_result", [SHA + "\n", "", "HEAD", "failure", "missing"])
@pytest.mark.parametrize(
    "contents",
    [None, "SECRET=private-value", "SECRET=private-value\nLANGSMITH_REVISION_ID=old\n"],
)
def test_build_stamps_revision(monkeypatch, tmp_path, capsys, git_result, contents):
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_SHA", FALLBACK_SHA)
    env = tmp_path / ".env"
    if contents is not None:
        env.write_text(contents)
    output = tmp_path / "build"

    def run(command, **kwargs):
        if command == ["git", "rev-parse", "HEAD"]:
            assert kwargs["cwd"] == tmp_path
            if git_result == "failure":
                raise subprocess.CalledProcessError(128, command)
            if git_result == "missing":
                raise FileNotFoundError("git")
            return subprocess.CompletedProcess(command, 0, stdout=git_result)
        output.mkdir()
        (output / "langgraph.json").write_text(json.dumps({"auth": {}}))

    monkeypatch.setattr(manage.subprocess, "run", run)
    config = json.loads(manage.build(output).read_text())
    values = dotenv_values(config["env"])
    assert values["LANGSMITH_REVISION_ID"] == (
        SHA if git_result == SHA + "\n" else FALLBACK_SHA
    )
    assert env.read_text().count("LANGSMITH_REVISION_ID=") == 1
    if contents is not None:
        assert values["SECRET"] == "private-value"
    captured = capsys.readouterr()
    assert "private-value" not in captured.out + captured.err


def test_build_fails_without_source_revision(monkeypatch, tmp_path):
    monkeypatch.setattr(manage, "ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(
        manage.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="HEAD"),
    )
    with pytest.raises(ValueError, match="Cannot resolve a git commit"):
        manage.build(tmp_path / "build")
    assert not (tmp_path / ".env").exists()
