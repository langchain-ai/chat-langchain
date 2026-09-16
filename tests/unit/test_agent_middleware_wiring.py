"""Tests for the managed agent middleware wiring."""

import os

for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.setdefault(key, "test-key")

from agent import docs_agent_middleware  # noqa: E402


def test_docs_agent_includes_guard_middleware():
    """Keep all documentation guard middleware in the managed stack."""
    middleware_names = {type(middleware).__name__ for middleware in docs_agent_middleware}

    assert middleware_names >= {
        "CitationGuardMiddleware",
        "DocsResearchGuardMiddleware",
        "DuplicateCallGuardMiddleware",
    }
