import pytest

from src.tools.source_code_tools import (
    _validate_path,
    _validate_repository,
)


def test_rejects_repository_outside_allowlist():
    with pytest.raises(ValueError, match="langchain"):
        _validate_repository("private-repo")


def test_accepts_allowlisted_repository_and_path():
    assert _validate_repository("langgraph") == "langgraph"
    assert _validate_path("libs/langgraph/langgraph/pregel/main.py") == (
        "libs/langgraph/langgraph/pregel/main.py"
    )


@pytest.mark.parametrize("path", ["../secrets.txt", "libs/../secrets.txt", "/"])
def test_rejects_unsafe_paths(path):
    with pytest.raises(ValueError):
        _validate_path(path)
