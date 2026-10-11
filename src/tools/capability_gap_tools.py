"""Tools for recording unsupported capability requests."""

from __future__ import annotations

from typing import Any

import langsmith as ls
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

CAPABILITY_GAP_LABELS = {
    "docs_export",
    "language_sdk",
    "deployment_data_access",
    "integration",
    "other",
}


def _root_run(runtime: ToolRuntime | None) -> tuple[Any, Any] | None:
    run_tree = ls.get_current_run_tree()
    if run_tree is not None:
        root = run_tree
        while getattr(root, "parent_run", None) is not None:
            root = root.parent_run
        root_id = getattr(run_tree, "trace_id", None) or root.id
        client = getattr(root, "client", None) or getattr(run_tree, "client", None)
        if root_id != root.id and client is not None and hasattr(client, "read_run"):
            root = client.read_run(root_id)
        return root, client

    if runtime is not None:
        run_id = runtime.config.get("run_id")
        client = runtime.config.get("client")
        if run_id and client is not None:
            return client.read_run(run_id), client
    return None


def _record_root_metadata(label: str, runtime: ToolRuntime | None) -> None:
    root_context = _root_run(runtime)
    if root_context is None:
        return
    root, client = root_context
    extra = dict(getattr(root, "extra", None) or {})
    metadata = dict(extra.get("metadata") or {})
    metadata["capability_gap"] = label
    extra["metadata"] = metadata
    root.extra = extra
    if client is not None:
        client.update_run(run_id=root.id, extra=extra)
    else:
        root.patch()


@tool
def record_capability_gap(
    label: str,
    user_request: str,
    runtime: ToolRuntime,
) -> str:
    """Record an unsupported capability using docs_export, language_sdk, deployment_data_access, integration, or other."""
    normalized_label = label if label in CAPABILITY_GAP_LABELS else "other"
    try:
        _record_root_metadata(normalized_label, runtime)
    except Exception:
        pass
    return "Capability gap recorded."


__all__ = ["record_capability_gap"]
