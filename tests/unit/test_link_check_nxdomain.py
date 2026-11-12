"""A hostname that does not exist is an invalid link, not an operational failure."""

import socket
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import src.tools.link_check_tools as lc


def _connect_error(errno):
    err = httpx.ConnectError("boom")
    err.__cause__ = socket.gaierror(errno, "resolver message")
    return err


def _client(error):
    client = MagicMock()
    client.head = AsyncMock(side_effect=error)
    client.get = AsyncMock(side_effect=error)
    return client


class TestNxdomain(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        lc._cache.clear()

    async def test_unknown_host_is_invalid_when_resolver_is_healthy(self):
        with patch.object(lc, "_resolver_is_healthy", return_value=True):
            result = await lc._check_single_url(
                _client(_connect_error(socket.EAI_NONAME)),
                "https://docs.illuminate.ai/oss/python/integrations/document_loaders",
                5.0,
            )
        self.assertFalse(result.valid)
        self.assertFalse(result.operational_failure)

    async def test_lookup_failure_is_operational_when_resolver_is_down(self):
        with patch.object(lc, "_resolver_is_healthy", return_value=False):
            result = await lc._check_single_url(
                _client(_connect_error(socket.EAI_NONAME)),
                "https://example-host.test/page",
                5.0,
            )
        self.assertFalse(result.valid)
        self.assertTrue(result.operational_failure)

    async def test_temporary_resolver_failure_stays_operational(self):
        with patch.object(lc, "_resolver_is_healthy", return_value=True):
            result = await lc._check_single_url(
                _client(_connect_error(socket.EAI_AGAIN)),
                "https://example-host.test/page",
                5.0,
            )
        self.assertTrue(result.operational_failure)

    async def test_connection_refused_stays_operational(self):
        with patch.object(lc, "_resolver_is_healthy", return_value=True):
            result = await lc._check_single_url(
                _client(httpx.ConnectError("refused")),
                "https://example-host.test/page",
                5.0,
            )
        self.assertTrue(result.operational_failure)


if __name__ == "__main__":
    unittest.main()
