"""Configure LangSmith tracing for the current deployment."""

from __future__ import annotations

import os

_PREVIEW_PROJECT_PREFIX = "engine-chat-langchain-pr-"
_HOST_PROJECT_ENV = "LANGSMITH_HOST_PROJECT_NAME"
_TRACE_PROJECT_ENV = "LANGSMITH_PROJECT"


def configure_langsmith_project() -> None:
    """Select the preview tracing project before models are initialized."""
    host_project = os.getenv(_HOST_PROJECT_ENV, "").strip()
    if not host_project.startswith(_PREVIEW_PROJECT_PREFIX):
        return

    os.environ[_TRACE_PROJECT_ENV] = os.getenv(_TRACE_PROJECT_ENV, "").strip() or host_project


def deployment_environment() -> str:
    """Return the trace environment for the current deployment."""
    host_project = os.getenv(_HOST_PROJECT_ENV, "").strip()
    return "production" if not host_project or host_project == "engine-chat-langchain" else "preview"


__all__ = ["configure_langsmith_project", "deployment_environment"]
