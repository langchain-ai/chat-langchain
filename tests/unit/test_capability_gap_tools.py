from types import SimpleNamespace
from uuid import uuid4

import langsmith

from src.tools.capability_gap_tools import record_capability_gap


def test_record_capability_gap_updates_root_metadata(monkeypatch):
    root_id = uuid4()
    client = SimpleNamespace(update_run=lambda **kwargs: updates.append(kwargs))
    updates = []
    root = SimpleNamespace(
        id=root_id,
        trace_id=root_id,
        extra={"metadata": {"existing": "value"}},
        client=client,
        patch=lambda: None,
    )
    monkeypatch.setattr(langsmith, "get_current_run_tree", lambda: root)

    result = record_capability_gap.func(
        "not-a-label", "Can I download the docs?", SimpleNamespace(config={})
    )

    assert result == "Capability gap recorded."
    assert updates == [
        {
            "run_id": root_id,
            "extra": {
                "metadata": {
                    "existing": "value",
                    "capability_gap": "other",
                }
            },
        }
    ]
