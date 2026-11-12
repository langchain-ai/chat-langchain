"""The link checker must never request loopback, private, link-local or metadata addresses."""

import socket
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

import src.tools.link_check_tools as lc


def _req(url):
    return httpx.Request("GET", url)


class TestPublicDestination(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        lc._cache.clear()

    async def test_blocks_literal_private_and_metadata_ips(self):
        for url in (
            "http://127.0.0.1/",
            "http://10.0.0.5/admin",
            "http://169.254.169.254/latest/meta-data/",
            "http://[::1]/",
            "http://192.168.1.1/",
        ):
            with self.assertRaises(lc.BlockedDestination, msg=url):
                await lc._assert_public_destination(_req(url))

    async def test_allows_literal_public_ip(self):
        await lc._assert_public_destination(_req("https://8.8.8.8/"))

    async def test_blocks_hostname_resolving_to_private_address(self):
        record = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 443))]
        loop = MagicMock()
        loop.getaddrinfo = AsyncMock(return_value=record)
        with patch("asyncio.get_running_loop", return_value=loop):
            with self.assertRaises(lc.BlockedDestination):
                await lc._assert_public_destination(_req("https://internal.example.com/"))

    async def test_allows_hostname_resolving_to_public_address(self):
        record = [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))]
        loop = MagicMock()
        loop.getaddrinfo = AsyncMock(return_value=record)
        with patch("asyncio.get_running_loop", return_value=loop):
            await lc._assert_public_destination(_req("https://example.com/"))

    async def test_failed_lookup_is_left_to_the_normal_error_path(self):
        loop = MagicMock()
        loop.getaddrinfo = AsyncMock(side_effect=socket.gaierror(socket.EAI_NONAME, "x"))
        with patch("asyncio.get_running_loop", return_value=loop):
            await lc._assert_public_destination(_req("https://no-such-host.test/"))

    async def test_check_single_url_marks_blocked_destination_invalid(self):
        client = MagicMock()
        client.head = AsyncMock(side_effect=lc.BlockedDestination("169.254.169.254"))
        result = await lc._check_single_url(client, "http://169.254.169.254/", 5.0)
        self.assertFalse(result.valid)
        self.assertFalse(result.operational_failure)


if __name__ == "__main__":
    unittest.main()
