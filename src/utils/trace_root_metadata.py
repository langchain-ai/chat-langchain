"""Root-run LangSmith metadata for the managed docs agent.

Pass this dict to ``define_deep_agent(metadata=...)`` so the runtime applies
``.with_config`` after ``create_deep_agent`` — nested middleware cannot
reliably reach the LangSmith root run tree.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import langsmith as ls
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse

from src.utils.prompt_provenance import get_prompt_provenance

_PROVENANCE_GRAPH_ID = "docs_agent"
_ENVIRONMENT_TEST_VALUES = {"dev", "development", "test", "testing", "staging"}


def _deployment_environment() -> str:
    """Return the bounded deployment environment label."""
    environment = os.getenv("LANGSMITH_ENV", "").strip().lower()
    project_name = os.getenv("LANGSMITH_HOST_PROJECT_NAME", "").strip().lower()
    if environment in _ENVIRONMENT_TEST_VALUES or project_name.endswith("-test"):
        return "test"
    return "production"


def _root_run_tree() -> Any | None:
    """Return the root run tree from the ambient tracing context."""
    run_tree = ls.get_current_run_tree()
    while run_tree is not None and run_tree.parent_run is not None:
        run_tree = run_tree.parent_run
    return run_tree


def _model_id(model: Any) -> str | None:
    """Return a bounded model identifier from a model or response."""
    for attribute in ("model_name", "model", "model_id"):
        value = model.get(attribute) if isinstance(model, Mapping) else getattr(model, attribute, None)
        if isinstance(value, str) and value:
            return value[:100]
    return None


def _same_model(first: str, second: str) -> bool:
    """Compare provider-qualified and unqualified model identifiers."""
    return first == second or first.rsplit(":", 1)[-1] == second.rsplit(":", 1)[-1]


class TraceOutcomeMiddleware(AgentMiddleware):
    """Record the model that successfully served the current turn."""

    def __init__(self, primary_model: str):
        """Initialize outcome tracking for the configured primary model."""
        super().__init__()
        self.primary_model = primary_model

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Record metadata after the successful model attempt."""
        response = handler(request)
        self._record_outcome(request, response)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Record metadata after the successful asynchronous model attempt."""
        response = await handler(request)
        self._record_outcome(request, response)
        return response

    def _record_outcome(self, request: ModelRequest, response: ModelResponse) -> None:
        """Write the successful model outcome to the root run."""
        served_model = None
        for message in reversed(response.result):
            metadata = getattr(message, "response_metadata", None) or {}
            served_model = _model_id(metadata)
            if served_model:
                break
        served_model = served_model or _model_id(request.model) or self.primary_model
        root_run = _root_run_tree()
        if root_run is not None:
            root_run.add_metadata(
                {
                    "model_served": served_model,
                    "model_fallback_used": not _same_model(
                        served_model, self.primary_model
                    ),
                }
            )


def build_docs_agent_trace_metadata(
    *,
    graph_id: str = _PROVENANCE_GRAPH_ID,
) -> dict[str, str]:
    """Return metadata that should land on the root LangSmith run."""
    metadata: dict[str, str] = {
        "source_type": "Chat-LangChain",
        "environment": _deployment_environment(),
        **get_prompt_provenance(graph_id),
    }
    revision = os.environ.get("LANGCHAIN_REVISION_ID") or os.environ.get(
        "LANGSMITH_HOST_REVISION_ID"
    )
    if revision:
        metadata["LANGSMITH_AGENT_VERSION"] = revision
    return metadata


__all__ = ["TraceOutcomeMiddleware", "build_docs_agent_trace_metadata"]
