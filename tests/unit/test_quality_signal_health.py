from datetime import datetime, timezone
from types import SimpleNamespace

from scripts.check_quality_signal_health import check_health


class FakeClient:
    def __init__(self, runs, feedback):
        self.runs = runs
        self.feedback = feedback

    def list_projects(self, *, name, limit):
        assert name == "docs_agent"
        assert limit == 2
        return iter([SimpleNamespace(id="project-id")])

    def list_runs(self, **kwargs):
        assert kwargs["project_id"] == "project-id"
        assert kwargs["is_root"] is True
        return iter(self.runs)

    def list_feedback(self, *, run_ids, feedback_key):
        assert run_ids == ["run-id"]
        return iter(self.feedback[feedback_key[0]])


def test_health_report_passes_with_scored_feedback():
    client = FakeClient(
        [
            SimpleNamespace(
                id="run-id",
                feedback_stats={"trajectory_accuracy": {"n": 1, "errors": 0}},
            )
        ],
        {
            "trajectory_accuracy": [SimpleNamespace(score=1, value=True)],
            "ux.thumb_vote": [SimpleNamespace(score=1, value="positive")],
        },
    )

    report = check_health(client, now=datetime.now(timezone.utc))

    assert report.failures == ()
    assert report.row_counts == {"trajectory_accuracy": 1, "ux.thumb_vote": 1}
    assert report.scored_counts["trajectory_accuracy"] == 1


def test_health_report_fails_on_errors_and_missing_feedback():
    client = FakeClient(
        [
            SimpleNamespace(
                id="run-id",
                feedback_stats={"trajectory_accuracy": {"n": 0, "errors": 1}},
            )
        ],
        {"trajectory_accuracy": [], "ux.thumb_vote": []},
    )

    report = check_health(client, now=datetime.now(timezone.utc))

    assert report.failures == (
        "trajectory_accuracy has 1 evaluator error(s)",
        "trajectory_accuracy has zero feedback rows",
        "ux.thumb_vote has zero feedback rows",
        "trajectory_accuracy has zero scored values",
    )
