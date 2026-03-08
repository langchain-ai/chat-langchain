"""Unit tests for OSS collection name consistency between system prompt and tool.

Bug: The system prompt listed "OSS" as the collection name for search_support_articles,
but the tool's actual collection (from the Pylon API) is named "OSS (LangChain and LangGraph)".
Because the matching logic does a case-insensitive exact match, "OSS" != "OSS (LangChain and
LangGraph)" and the agent received a collection-not-found error instead of OSS articles.

Fix: Updated docs_agent_prompt.py to use "OSS (LangChain and LangGraph)" everywhere.
"""

import json
import re
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers to parse collection names from the system prompt
# ---------------------------------------------------------------------------

PROMPT_FILE = "src/prompts/docs_agent_prompt.py"
PYLON_TOOLS_FILE = "src/tools/pylon_tools.py"


def _load_prompt_text() -> str:
    """Load the raw text of the system-prompt module."""
    with open(PROMPT_FILE, encoding="utf-8") as fh:
        return fh.read()


def _load_pylon_tools_text() -> str:
    """Load the raw text of pylon_tools."""
    with open(PYLON_TOOLS_FILE, encoding="utf-8") as fh:
        return fh.read()


def _extract_prompt_collection_names(text: str) -> list[str]:
    """Return all quoted collection names listed under the search_support_articles
    section of the system prompt.

    The section looks like:
        - "CollectionName" - description
    """
    # Find the search_support_articles section
    section_match = re.search(
        r"search_support_articles.*?(?=###|\Z)", text, re.DOTALL
    )
    if not section_match:
        return []
    section = section_match.group(0)
    # Extract every quoted name that appears as a bullet-list item
    return re.findall(r'^\s*-\s+"([^"]+)"', section, re.MULTILINE)


def _extract_docstring_collection_names(text: str) -> list[str]:
    """Return all quoted collection names from the search_support_articles docstring."""
    # Grab the docstring of search_support_articles
    ds_match = re.search(
        r'def search_support_articles.*?"""(.*?)"""', text, re.DOTALL
    )
    if not ds_match:
        return []
    docstring = ds_match.group(1)
    return re.findall(r'^\s*-\s+"([^"]+)"', docstring, re.MULTILINE)


# ---------------------------------------------------------------------------
# Tests: collection-name consistency
# ---------------------------------------------------------------------------


class TestOSSCollectionNameConsistency(unittest.TestCase):
    """Verify that the system prompt uses the same collection names as the tool."""

    def setUp(self):
        self.prompt_text = _load_prompt_text()
        self.tools_text = _load_pylon_tools_text()
        self.prompt_collections = _extract_prompt_collection_names(self.prompt_text)
        self.docstring_collections = _extract_docstring_collection_names(
            self.tools_text
        )

    def test_prompt_collections_parsed(self):
        """Sanity check: we can extract at least one collection from the prompt."""
        self.assertTrue(
            len(self.prompt_collections) > 0,
            "No collection names found in the system prompt – check the regex.",
        )

    def test_docstring_collections_parsed(self):
        """Sanity check: we can extract at least one collection from the tool docstring."""
        self.assertTrue(
            len(self.docstring_collections) > 0,
            "No collection names found in the tool docstring – check the regex.",
        )

    def test_oss_collection_name_not_bare_in_prompt(self):
        """The bare name 'OSS' (the old, broken value) must NOT appear as a collection
        entry in the system prompt."""
        self.assertNotIn(
            "OSS",
            self.prompt_collections,
            "System prompt still uses the bare 'OSS' collection name. "
            "It must be 'OSS (LangChain and LangGraph)' to match the tool.",
        )

    def test_oss_full_name_present_in_prompt(self):
        """The correct full name must appear in the system prompt collection list."""
        self.assertIn(
            "OSS (LangChain and LangGraph)",
            self.prompt_collections,
            "System prompt is missing 'OSS (LangChain and LangGraph)' collection entry.",
        )

    def test_oss_full_name_present_in_docstring(self):
        """The full name must also be present in the tool docstring (source of truth)."""
        self.assertIn(
            "OSS (LangChain and LangGraph)",
            self.docstring_collections,
            "Tool docstring is missing 'OSS (LangChain and LangGraph)' collection entry.",
        )

    def test_all_prompt_collections_exist_in_docstring(self):
        """Every collection name advertised in the system prompt must also appear in the
        tool docstring so there are no phantom names that would produce a not-found error."""
        docstring_set = set(self.docstring_collections)
        for name in self.prompt_collections:
            self.assertIn(
                name,
                docstring_set,
                f"Collection '{name}' is in the system prompt but NOT in the tool "
                f"docstring. Available in docstring: {sorted(docstring_set)}",
            )


# ---------------------------------------------------------------------------
# Tests: collection-matching logic (mocked API)
# ---------------------------------------------------------------------------


def _simulate_collection_match(requested: str, available: dict[str, str]) -> bool:
    """Mirror the matching logic from pylon_tools.search_support_articles.

    Returns True if the requested collection name resolves to a known ID.
    """
    if requested in available:
        return True
    # Case-insensitive fallback
    for key in available:
        if key.lower() == requested.lower():
            return True
    return False


MOCK_COLLECTION_MAP = {
    "General": "col-001",
    "OSS (LangChain and LangGraph)": "col-002",
    "LangSmith Observability": "col-003",
    "LangSmith Evaluation": "col-004",
    "LangSmith Deployment": "col-005",
    "SDKs and APIs": "col-006",
    "LangSmith Studio": "col-007",
    "Self Hosted": "col-008",
    "Troubleshooting": "col-009",
    "Security": "col-010",
}


class TestCollectionMatchingLogic(unittest.TestCase):
    """Unit-test the collection-name matching logic in isolation."""

    def test_bare_oss_does_not_match(self):
        """Confirm that the OLD broken value 'OSS' does NOT match any real collection."""
        self.assertFalse(
            _simulate_collection_match("OSS", MOCK_COLLECTION_MAP),
            "'OSS' should NOT match any collection – it is the old broken name.",
        )

    def test_full_oss_name_matches_exactly(self):
        """The correct full name resolves correctly with an exact match."""
        self.assertTrue(
            _simulate_collection_match(
                "OSS (LangChain and LangGraph)", MOCK_COLLECTION_MAP
            ),
            "'OSS (LangChain and LangGraph)' must match the collection map.",
        )

    def test_full_oss_name_matches_case_insensitively(self):
        """Case-insensitive match also works for the full name."""
        self.assertTrue(
            _simulate_collection_match(
                "oss (langchain and langgraph)", MOCK_COLLECTION_MAP
            ),
            "Case-insensitive match must work for 'OSS (LangChain and LangGraph)'.",
        )

    def test_all_prompt_collections_match(self):
        """Every collection name in the (fixed) system prompt must resolve in the mock map."""
        prompt_text = _load_prompt_text()
        prompt_collections = _extract_prompt_collection_names(prompt_text)
        # "all" is a special keyword, not a collection name
        for name in prompt_collections:
            self.assertTrue(
                _simulate_collection_match(name, MOCK_COLLECTION_MAP),
                f"Collection '{name}' from the system prompt does NOT match any "
                f"entry in the mock collection map. This would cause a not-found error.",
            )


# ---------------------------------------------------------------------------
# Tests: search_support_articles returns error for bare "OSS" (mocked API)
# ---------------------------------------------------------------------------


class TestSearchSupportArticlesIntegration(unittest.TestCase):
    """Integration-style tests for search_support_articles with mocked HTTP calls."""

    def _make_mock_articles(self) -> list[dict]:
        return [
            {
                "is_published": True,
                "title": "Getting started with LangGraph",
                "visibility_config": {"visibility": "public"},
                "identifier": "abc123",
                "slug": "getting-started-langgraph",
                "id": "art-001",
                "collection_id": "col-002",
            }
        ]

    def _make_mock_collections_response(self) -> dict:
        return {
            "data": [
                {
                    "title": name,
                    "id": cid,
                    "visibility_config": {"visibility": "public"},
                }
                for name, cid in MOCK_COLLECTION_MAP.items()
            ]
        }

    def _make_mock_articles_response(self) -> dict:
        return {"data": self._make_mock_articles()}

    @patch("src.tools.pylon_tools._fetch_collections")
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_bare_oss_returns_error(self, mock_articles, mock_collections):
        """Calling search_support_articles with the OLD 'OSS' name returns an error JSON."""
        mock_articles.return_value = self._make_mock_articles()
        mock_collections.return_value = self._make_mock_collections_response()["data"]

        from src.tools.pylon_tools import search_support_articles

        result = search_support_articles.invoke({"collections": "OSS"})
        result_data = json.loads(result)
        self.assertIn("error", result_data)
        self.assertIn("OSS", result_data["error"])

    @patch("src.tools.pylon_tools._fetch_collections")
    @patch("src.tools.pylon_tools._fetch_all_articles")
    def test_full_oss_name_returns_articles(self, mock_articles, mock_collections):
        """Calling search_support_articles with the CORRECT full name returns articles."""
        mock_articles.return_value = self._make_mock_articles()
        mock_collections.return_value = self._make_mock_collections_response()["data"]

        from src.tools.pylon_tools import search_support_articles

        result = search_support_articles.invoke(
            {"collections": "OSS (LangChain and LangGraph)"}
        )
        result_data = json.loads(result)
        self.assertNotIn("error", result_data)
        self.assertIn("articles", result_data)
        self.assertGreater(len(result_data["articles"]), 0)
