"""Tests for `_is_safe_destination` SSRF guard in link_check_tools."""

from unittest.mock import patch

import pytest

from src.tools.link_check_tools import _is_safe_destination


def _fake_getaddrinfo_factory(ip: str):
    """Return a fake getaddrinfo that always resolves to *ip*."""
    def _fake(host, port, *args, **kwargs):  # noqa: ARG001
        # mirror the (family, type, proto, canonname, sockaddr) tuple shape
        return [(0, 0, 0, "", (ip, port or 0))]

    return _fake


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:5432",
        "http://localhost.localdomain/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://127.0.0.1",
        "http://127.0.0.1:8080/internal",
        "http://192.168.0.1/admin",
        "http://10.0.0.1",
        "http://172.16.0.1",
        "http://169.254.169.254",
        "http://0.0.0.0",
        "http://[::1]/",
        "http://[fe80::1]/",
        "file:///etc/passwd",
        "ftp://example.com",
    ],
)
def test_blocks_private_and_disallowed_urls(url):
    # Stub DNS so the hostname-based cases (localhost, metadata.google.internal)
    # don't escape to a real resolver if the literal-IP short-circuit misses.
    with patch(
        "src.tools.link_check_tools.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo_factory("127.0.0.1"),
    ):
        assert _is_safe_destination(url) is False


@pytest.mark.parametrize(
    "url,resolved_ip",
    [
        ("http://docs.langchain.com", "104.18.32.7"),
        ("https://docs.langchain.com/intro", "104.18.32.7"),
        ("http://github.com", "140.82.114.4"),
    ],
)
def test_allows_public_destinations(url, resolved_ip):
    with patch(
        "src.tools.link_check_tools.socket.getaddrinfo",
        side_effect=_fake_getaddrinfo_factory(resolved_ip),
    ):
        assert _is_safe_destination(url) is True


def test_fails_closed_when_resolution_raises():
    with patch(
        "src.tools.link_check_tools.socket.getaddrinfo",
        side_effect=OSError("dns failure"),
    ):
        assert _is_safe_destination("http://docs.langchain.com") is False


def test_blocks_when_any_resolved_address_is_private():
    def _mixed(host, port, *args, **kwargs):  # noqa: ARG001
        return [
            (0, 0, 0, "", ("104.18.32.7", 0)),
            (0, 0, 0, "", ("10.0.0.5", 0)),
        ]

    with patch("src.tools.link_check_tools.socket.getaddrinfo", side_effect=_mixed):
        assert _is_safe_destination("http://attacker.example.com") is False
