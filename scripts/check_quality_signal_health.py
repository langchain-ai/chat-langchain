"""Check recent LangSmith feedback health for the docs agent project."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from langsmith import Client

PROJECT_NAME = "docs_agent"
TRAJECTORY_FEEDBACK_KEY = "trajectory_accuracy"
THUMB_FEEDBACK_KEY = "ux.thumb_vote"
FEEDBACK_KEYS = (TRAJECTORY_FEEDBACK_KEY, THUMB_FEEDBACK_KEY)


@dataclass(frozen=True)
class HealthReport:
    """Summarize feedback health-check failures."""

    row_counts: dict[str, int]
    scored_counts: dict[str, int]
    evaluator_errors: int
    failures: tuple[str, ...]


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _as_count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _feedback_stat(run: Any, key: str) -> Mapping[str, Any]:
    stats = _field(run, "feedback_stats", {}) or {}
    stat = stats.get(key, {}) if isinstance(stats, Mapping) else {}
    return stat if isinstance(stat, Mapping) else {}


def _feedback_error_count(feedback: Any) -> int:
    direct_error = _field(feedback, "error")
    if direct_error:
        return _as_count(direct_error) or 1
    extra = _field(feedback, "extra", {}) or {}
    if isinstance(extra, Mapping):
        return _as_count(extra.get("errors") or extra.get("error"))
    return 0


def _has_scored_value(feedback: Any) -> bool:
    score = _field(feedback, "score")
    value = _field(feedback, "value")
    return score is not None or value not in (None, "")


def _project_id(client: Client, project_name: str) -> str:
    configured_id = os.getenv("LANGSMITH_PROJECT_ID")
    if configured_id:
        return configured_id
    projects = list(client.list_projects(name=project_name, limit=2))
    if len(projects) != 1 or projects[0].id is None:
        raise RuntimeError(f"Expected one LangSmith project named {project_name!r}")
    return str(projects[0].id)


def _list_feedback(client: Client, run_ids: list[str], key: str) -> list[Any]:
    if not run_ids:
        return []
    feedback: list[Any] = []
    for offset in range(0, len(run_ids), 100):
        feedback.extend(
            client.list_feedback(
                run_ids=run_ids[offset : offset + 100], feedback_key=[key]
            )
        )
    return feedback


def check_health(
    client: Client,
    *,
    project_name: str = PROJECT_NAME,
    now: datetime | None = None,
) -> HealthReport:
    """Check feedback rows, scores, and evaluator errors from the last 24 hours."""
    current_time = now or datetime.now(UTC)
    start_time = current_time - timedelta(hours=24)
    project_id = _project_id(client, project_name)
    runs = list(
        client.list_runs(
            project_id=project_id,
            start_time=start_time,
            is_root=True,
            select=["id", "feedback_stats"],
        )
    )
    run_ids = [str(run_id) for run in runs if (run_id := _field(run, "id"))]

    feedback_by_key = {
        key: _list_feedback(client, run_ids, key) for key in FEEDBACK_KEYS
    }
    row_counts = {key: len(feedback_by_key[key]) for key in FEEDBACK_KEYS}
    scored_counts = {
        key: sum(_has_scored_value(feedback) for feedback in feedback_by_key[key])
        for key in FEEDBACK_KEYS
    }

    evaluator_errors = sum(
        _as_count(_feedback_stat(run, TRAJECTORY_FEEDBACK_KEY).get("errors"))
        for run in runs
    )
    evaluator_errors += sum(
        _feedback_error_count(feedback)
        for feedback in feedback_by_key[TRAJECTORY_FEEDBACK_KEY]
    )
    trajectory_stats_scored = sum(
        _as_count(_feedback_stat(run, TRAJECTORY_FEEDBACK_KEY).get("n")) for run in runs
    )
    trajectory_scored = max(
        scored_counts[TRAJECTORY_FEEDBACK_KEY], trajectory_stats_scored
    )

    failures: list[str] = []
    if evaluator_errors:
        failures.append(
            f"{TRAJECTORY_FEEDBACK_KEY} has {evaluator_errors} evaluator error(s)"
        )
    for key in FEEDBACK_KEYS:
        if row_counts[key] == 0:
            failures.append(f"{key} has zero feedback rows")
    if trajectory_scored == 0:
        failures.append(f"{TRAJECTORY_FEEDBACK_KEY} has zero scored values")

    return HealthReport(
        row_counts=row_counts,
        scored_counts={
            **scored_counts,
            TRAJECTORY_FEEDBACK_KEY: trajectory_scored,
        },
        evaluator_errors=evaluator_errors,
        failures=tuple(failures),
    )


def main() -> int:
    """Run the health check and return a process exit code."""
    report = check_health(
        Client(),
        project_name=os.getenv("LANGSMITH_PROJECT", PROJECT_NAME),
    )
    sys.stdout.write(f"{json.dumps(report.__dict__, sort_keys=True)}\n")
    if report.failures:
        for failure in report.failures:
            sys.stderr.write(f"quality signal check failed: {failure}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
