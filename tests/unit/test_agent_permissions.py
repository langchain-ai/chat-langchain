"""Tests for the docs agent filesystem permissions."""

from pathlib import Path


def test_agent_declares_scoped_filesystem_permissions():
    source = Path("agent.py").read_text()

    assert (
        'FilesystemPermission(operations=["read"], paths=["/large_tool_results/**"])'
        in source
    )
    assert (
        'FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny")'
        in source
    )
    assert "permissions=docs_agent_permissions" in source
