"""Helpers for recording dynamic metadata on the LangSmith root run."""

from __future__ import annotations

import logging

from langsmith.run_helpers import get_current_run_tree

logger = logging.getLogger(__name__)


def set_root_metadata(**values: str) -> None:
    """Merge values into the current LangSmith root run metadata."""
    run_tree = get_current_run_tree()
    if run_tree is None:
        logger.debug("No current LangSmith run tree; skipping root metadata")
        return

    root_run = getattr(run_tree, "root", None) or run_tree
    while getattr(root_run, "parent_run", None) is not None:
        root_run = root_run.parent_run

    metadata = root_run.extra.setdefault("metadata", {})
    metadata.update(values)
