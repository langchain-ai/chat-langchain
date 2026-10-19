"""Build and deploy the managed agent with the public chat authorization policy."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".mda" / "build"


def preflight() -> None:
    """Verify the deployment environment can access the Pylon knowledge base."""
    try:
        lines = (ROOT / ".env").read_text().splitlines()
    except OSError:
        raise SystemExit("Deployment aborted: cannot read project .env.") from None
    environment = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        environment[key.strip()] = value
    for key in ("PYLON_API_KEY", "PYLON_KB_ID"):
        if not environment.get(key):
            raise SystemExit(f"Deployment aborted: {key} is required in project .env.")
    try:
        response = requests.get(
            "https://api.usepylon.com/knowledge-bases/"
            f"{environment['PYLON_KB_ID']}/articles",
            headers={"Authorization": f"Bearer {environment['PYLON_API_KEY']}"},
            timeout=10,
            allow_redirects=False,
        )
    except requests.RequestException:
        raise SystemExit(
            "Deployment aborted: Pylon preflight request failed."
        ) from None
    if response.status_code in (401, 403):
        raise SystemExit(
            f"Deployment aborted: Pylon returned HTTP {response.status_code}; "
            "rotate PYLON_API_KEY and verify PYLON_KB_ID in project .env."
        )
    if response.status_code != 200:
        raise SystemExit(
            f"Deployment aborted: Pylon preflight returned HTTP {response.status_code}."
        )


def build(output: Path = BUILD) -> Path:
    """Compile MDA artifacts and select the maintained authorization handler."""
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
    if args.command == "deploy":
        preflight()
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
