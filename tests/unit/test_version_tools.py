"""Tests for get_package_version() in src/tools/version_tools.py.

These tests do NOT require network access or LangSmith credentials.
All HTTP calls are mocked via unittest.mock.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import src.tools.version_tools as version_module
from src.tools.version_tools import get_package_version


@pytest.fixture(autouse=True)
def _clear_cache():
    """Reset the in-process TTL cache before each test."""
    version_module._cache.clear()
    yield
    version_module._cache.clear()


def _patch_client(response=None, side_effect=None):
    """Patch httpx.AsyncClient so client.get() returns *response* or raises."""
    client = MagicMock()
    client.get = AsyncMock(return_value=response, side_effect=side_effect)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return patch(
        "src.tools.version_tools.httpx.AsyncClient", return_value=client
    ), client


def _make_response(payload):
    """Build a mock httpx.Response whose .json() returns *payload*."""
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


@pytest.mark.asyncio
async def test_pypi_returns_info_version():
    resp = _make_response({"info": {"version": "1.2.3"}})
    patcher, client = _patch_client(response=resp)
    with patcher:
        result = await get_package_version.ainvoke(
            {"package": "langgraph", "ecosystem": "pypi"}
        )

    assert "1.2.3" in result
    assert client.get.await_args[0][0] == "https://pypi.org/pypi/langgraph/json"


@pytest.mark.asyncio
async def test_npm_returns_dist_tags_latest():
    resp = _make_response({"dist-tags": {"latest": "0.4.9"}})
    patcher, client = _patch_client(response=resp)
    with patcher:
        result = await get_package_version.ainvoke(
            {"package": "@langchain/core", "ecosystem": "npm"}
        )

    assert "0.4.9" in result
    assert client.get.await_args[0][0] == "https://registry.npmjs.org/@langchain/core"


@pytest.mark.asyncio
async def test_non_allowlisted_package_makes_no_http_call():
    patcher, client = _patch_client(response=_make_response({}))
    with patcher:
        result = await get_package_version.ainvoke(
            {"package": "requests", "ecosystem": "pypi"}
        )

    assert "not a LangChain ecosystem package" in result
    client.get.assert_not_awaited()


@pytest.mark.asyncio
async def test_network_error_returns_message_without_raising():
    patcher, _ = _patch_client(side_effect=httpx.ConnectError("boom"))
    with patcher:
        result = await get_package_version.ainvoke(
            {"package": "langchain", "ecosystem": "pypi"}
        )

    assert result == "Could not resolve the current version of langchain (pypi)."


@pytest.mark.asyncio
async def test_http_error_status_returns_message_without_raising():
    resp = MagicMock()
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "404", request=MagicMock(), response=MagicMock(status_code=404)
    )
    patcher, _ = _patch_client(response=resp)
    with patcher:
        result = await get_package_version.ainvoke(
            {"package": "langsmith", "ecosystem": "pypi"}
        )

    assert result == "Could not resolve the current version of langsmith (pypi)."
