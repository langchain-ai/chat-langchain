"""Deployment environment helpers."""

import os


def is_staging_environment() -> bool:
    """Return whether the deployment is configured as staging."""
    return (
        os.getenv("LANGSMITH_HOST_PROJECT_NAME") == "immanuel-chat-langchain-test"
        or os.getenv("LANGSMITH_ENV") == "dev"
    )


def get_environment() -> str:
    """Return the deployment environment label."""
    return "staging" if is_staging_environment() else "production"


__all__ = ["get_environment", "is_staging_environment"]
