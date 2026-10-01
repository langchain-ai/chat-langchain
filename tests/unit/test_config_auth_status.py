import os
import subprocess
import sys


def test_missing_primary_key_exports_auth_failure_status():
    environment = {
        **os.environ,
        "OPENAI_API_KEY": "test-key",
        "ANTHROPIC_API_KEY": "test-key",
        "PRIMARY_MODEL_AUTH_VALIDATION_SKIP": "1",
    }
    environment.pop("GOOGLE_API_KEY", None)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.agent import config; assert not config.PRIMARY_MODEL_AUTH_OK",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
