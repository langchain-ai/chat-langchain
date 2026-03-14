"""Regression tests for SearchDocsByLangChain URL construction.

Verifies that _format_search_results correctly constructs URLs with a '/'
separator between the domain and path, using exact paths observed in
production traces from 2026-03-14 that exhibited broken URLs.

Broken URLs seen in traces (after commit #512 was merged on 2026-03-10):
- https://docs.langchain.comlangsmith/cli
- https://docs.langchain.comoss/javascript/langchain/multi-agent/router
- https://docs.langchain.comapi-reference/integrations-v1/list-github-integrations

Root cause confirmed: deployed SHA a6667a67 predates the fix; the code path is
  f"https://docs.langchain.com{path}"  (no slash separator)
Fix (src/tools/docs_tools.py line 222):
  f"https://docs.langchain.com/{path.lstrip('/')}"
"""

from unittest.mock import patch

import pytest

from src.tools.docs_tools import _format_search_results

BASE_URL = "https://docs.langchain.com"


def _make_result(path: str, title: str = "Test Page", content: str = "body") -> dict:
    return {"path": path, "metadata": {"title": title}, "content": content}


def _extract_url(output: str) -> str:
    for line in output.splitlines():
        if line.startswith("Link:"):
            return line.split("Link:", 1)[1].strip()
    raise ValueError(f"No 'Link:' line in output:\n{output}")


# ---------------------------------------------------------------------------
# Paths extracted directly from the three failing production traces
# ---------------------------------------------------------------------------


class TestProductionTracePaths:
    """Reproduce the exact paths seen in the 2026-03-14 broken-URL traces."""

    @pytest.mark.parametrize(
        ("path", "expected_url"),
        [
            (
                "langsmith/cli",
                "https://docs.langchain.com/langsmith/cli",
            ),
            (
                "oss/javascript/langchain/multi-agent/router",
                "https://docs.langchain.com/oss/javascript/langchain/multi-agent/router",
            ),
            (
                "api-reference/integrations-v1/list-github-integrations",
                "https://docs.langchain.com/api-reference/integrations-v1/list-github-integrations",
            ),
            (
                "oss/javascript/deepagents/cli/providers",
                "https://docs.langchain.com/oss/javascript/deepagents/cli/providers",
            ),
        ],
    )
    def test_production_path_produces_valid_url(self, path, expected_url):
        """Each path from a real broken-URL trace must yield a properly formed URL."""
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(path)])
        url = _extract_url(output)
        assert url == expected_url, (
            f"Expected '{expected_url}', got '{url}'. "
            "The slash-separator bug is still present for this path."
        )

    @pytest.mark.parametrize(
        "path",
        [
            "langsmith/cli",
            "oss/javascript/langchain/multi-agent/router",
            "api-reference/integrations-v1/list-github-integrations",
            "oss/javascript/deepagents/cli/providers",
        ],
    )
    def test_no_domain_path_merge(self, path):
        """URL must never merge domain and path without a slash (the original bug)."""
        with patch("src.tools.docs_tools._track_docs_for_langsmith"):
            output = _format_search_results([_make_result(path)])
        url = _extract_url(output)
        merged = f"https://docs.langchain.com{path}"
        assert url != merged, (
            f"Broken URL '{merged}' produced — slash separator is missing."
        )
        assert url.startswith(f"{BASE_URL}/"), (
            f"URL '{url}' does not start with '{BASE_URL}/'."
        )
