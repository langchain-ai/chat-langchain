"""Tests for link and documentation filesystem target guards."""

from unittest.mock import AsyncMock

import pytest

from src.middleware.docs_filesystem_guard_middleware import (
    _is_filesystem_command_permitted,
)
from src.tools.link_check_tools import _check_single_url, _is_permitted_target


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:65534",
        "http://10.0.0.5",
        "http://169.254.169.254/latest/meta-data/",
        "https://langchain-docs.example/agent-guide",
    ],
)
def test_is_permitted_target_rejects_untrusted_targets(url, monkeypatch):
    monkeypatch.setattr("src.tools.link_check_tools.socket.getaddrinfo", lambda *args: [])
    assert not _is_permitted_target(url)


def test_is_permitted_target_accepts_docs_host(monkeypatch):
    monkeypatch.setattr(
        "src.tools.link_check_tools.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("8.8.8.8", 0))],
    )
    assert _is_permitted_target("https://docs.langchain.com/oss/python/langchain/agents")


@pytest.mark.asyncio
async def test_unpermitted_redirect_is_not_followed(monkeypatch):
    response = type(
        "Response",
        (),
        {"status_code": 302, "headers": {"location": "http://127.0.0.1:65534"}},
    )()
    client = type("Client", (), {"head": AsyncMock(return_value=response)})()
    monkeypatch.setattr(
        "src.tools.link_check_tools.socket.getaddrinfo",
        lambda *args, **kwargs: [(None, None, None, None, ("8.8.8.8", 0))],
    )

    result = await _check_single_url(client, "https://www.langchain.com/redirect", 1.0)

    assert not result.valid
    assert "Target not permitted" in result.error
    client.head.assert_awaited_once()


@pytest.mark.parametrize(
    "command",
    [
        "head -5 /etc/hostname",
        "head -5 ../etc/hostname",
        "head -5 $(cat /etc/hostname)",
    ],
)
def test_filesystem_guard_rejects_unsafe_commands(command):
    assert not _is_filesystem_command_permitted(command)


def test_filesystem_guard_accepts_documentation_read():
    assert _is_filesystem_command_permitted("head -100 /oss/python/langchain/agents.mdx")
