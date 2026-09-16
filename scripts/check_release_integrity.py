"""Verify the production revision is part of the default branch history."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _deployment_revision() -> str:
    """Read the serving revision from the LangSmith deployment API."""
    endpoint = os.getenv(
        "LANGSMITH_CONTROL_PLANE_URL", "https://api.host.langchain.com"
    ).rstrip("/")
    name = os.environ["LANGSMITH_DEPLOYMENT_NAME"]
    query = urlencode({"name_contains": name})
    request = Request(f"{endpoint}/v2/deployments/?{query}")
    request.add_header("X-Api-Key", os.environ["LANGSMITH_API_KEY"])
    tenant_id = os.getenv("LANGSMITH_TENANT_ID")
    if tenant_id:
        request.add_header("X-Tenant-Id", tenant_id)
    with urlopen(request) as response:
        payload = json.load(response)
    deployments = payload.get("deployments", payload) if isinstance(payload, dict) else payload
    if isinstance(deployments, dict):
        deployments = [deployments]
    for deployment in deployments:
        if deployment.get("name") != name and len(deployments) != 1:
            continue
        for field in ("revision_id", "latest_revision_id", "source_revision"):
            revision = deployment.get(field)
            if revision:
                return revision
        source_config = deployment.get("source_config", {})
        for field in ("revision_id", "source_revision"):
            revision = source_config.get(field)
            if revision:
                return revision
    raise RuntimeError(f"No serving revision found for deployment {name}")


def _default_branch() -> str:
    """Return the checked-out repository's default branch name."""
    try:
        ref = subprocess.check_output(
            ["git", "symbolic-ref", "refs/remotes/origin/HEAD"], text=True
        ).strip()
        return ref.rsplit("/", 1)[-1]
    except (OSError, subprocess.CalledProcessError):
        return "main"


def main() -> None:
    """Fail if production serves a revision outside the default branch."""
    serving = _deployment_revision()
    branch = _default_branch()
    head = subprocess.check_output(
        ["git", "rev-parse", f"origin/{branch}"], text=True
    ).strip()
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", serving, head], check=False
    )
    if result.returncode != 0:
        commits = subprocess.check_output(
            ["git", "log", "--oneline", f"{serving}..{head}"], text=True
        ).strip()
        raise SystemExit(
            f"Serving revision {serving} is not an ancestor of origin/{branch} {head}.\n"
            f"Merged-but-undeployed commits:\n{commits}"
        )
    sys.stdout.write(
        f"Serving revision {serving} is an ancestor of origin/{branch} {head}.\n"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write(f"Release integrity check failed: {exc}\n")
        raise
