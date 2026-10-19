"""Build and deploy the managed agent with the public chat authorization policy."""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from dotenv import set_key

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".mda" / "build"


def resolve_revision() -> str:
    """Resolve the source commit from git or the CI checkout SHA."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        revision = result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = ""
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision):
        revision = os.environ.get("GITHUB_SHA", "").strip()
    if not re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", revision):
        raise ValueError("Cannot resolve a git commit from HEAD or GITHUB_SHA")
    return revision


def build(output: Path = BUILD) -> Path:
    """Compile MDA artifacts and select the maintained authorization handler."""
    set_key(
        ROOT / ".env", "LANGSMITH_REVISION_ID", resolve_revision(), quote_mode="never"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "managed_deepagents",
            "build",
            str(ROOT),
            "--out",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )
    config_path = output / "langgraph.json"
    config = json.loads(config_path.read_text())
    config["auth"]["path"] = "./auth.py:auth"
    config["env"] = str(ROOT / ".env")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def main() -> None:
    """Build the secured bundle or deploy it without invoking MDA regeneration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "deploy"])
    args, deployment_args = parser.parse_known_args()
    if args.command == "build" and deployment_args:
        parser.error("build does not accept deployment arguments")
    config_path = build()
    if args.command == "deploy":
        subprocess.run(
            [
                sys.executable,
                "-m",
                "langgraph_cli",
                "deploy",
                "--config",
                str(config_path),
                *deployment_args,
            ],
            check=True,
            cwd=ROOT,
        )


if __name__ == "__main__":
    main()
