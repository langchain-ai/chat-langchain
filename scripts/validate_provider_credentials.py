"""Validate provider credentials before deployment."""

import os

from src.agent.config import validate_provider_credentials


def _enabled(value: str | None) -> bool:
    return value not in {None, "", "0", "false", "False", "no", "No"}


if _enabled(os.getenv("VALIDATE_PROVIDER_CREDENTIALS")):
    validate_provider_credentials(fatal=True)
