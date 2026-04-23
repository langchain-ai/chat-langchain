"""Tests for guardrails false positives on DeepAgents and LangSmith terms.

These tests verify that the guardrails system prompt explicitly enumerates
DeepAgents-specific terms (Skills, FilesystemMiddleware, etc.) and LangSmith
feature names (Annotation Queues, Datasets, etc.) in the ALWAYS ALLOW section,
preventing the classifier from blocking them.

Production traces showing false positives:
- 019cfa70-7003: "Annotation Queues 的数据表是什么样的？" — BLOCKED TWICE
- 019cf6a3-eb71: "What are skills?" — BLOCKED
- 019cf718-0c52: "skills가 뭐야" (Korean: "what are skills?") — BLOCKED TWICE
- 019cf8d9-0391: "FilesystemMiddleware opowiedz mi o tym" — BLOCKED
- 019cd791-bde1: "FilesystemMiddleware 的用法" — BLOCKED
- 019a4aff-25f4: "What are deep agents?" — BLOCKED
- 019ab8e8-c975: "deepagents 的主要特征有那些" — BLOCKED TWICE
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.middleware.guardrails_middleware import _GUARDRAILS_SYSTEM_PROMPT

PROMPT_LOWER = _GUARDRAILS_SYSTEM_PROMPT.lower()


def _get_always_allow_section() -> str:
    """Extract text from '## always allow' through '## only block' (exclusive).

    Uses the section-header form '## only block' to avoid matching the
    phrase 'only block when you are HIGHLY CONFIDENT' that appears earlier
    in the preamble.
    """
    always_allow_idx = PROMPT_LOWER.find("## always allow")
    assert always_allow_idx != -1, "Prompt must have a '## ALWAYS ALLOW' section"

    only_block_idx = PROMPT_LOWER.find("## only block")
    assert only_block_idx != -1, "Prompt must have a '## ONLY BLOCK' section"

    assert always_allow_idx < only_block_idx, (
        f"'## ALWAYS ALLOW' (pos {always_allow_idx}) must appear before "
        f"'## ONLY BLOCK' (pos {only_block_idx}) in the prompt."
    )

    return PROMPT_LOWER[always_allow_idx:only_block_idx]


# ---------------------------------------------------------------------------
# DeepAgents-specific term tests
# ---------------------------------------------------------------------------


def test_guardrails_prompt_mentions_skills_as_allowed():
    """The guardrails prompt must mention 'skills' in the ALWAYS ALLOW section.

    Production traces show the classifier blocks "What are skills?" and "skills가 뭐야"
    because it doesn't know "Skills" is a DeepAgents concept.

    Failing traces:
      - 019cf6a3: "What are skills?" — BLOCKED
      - 019cf718: "skills가 뭐야" — BLOCKED TWICE
    """
    allow_section = _get_always_allow_section()
    assert "skills" in allow_section, (
        "The guardrails ALWAYS ALLOW section must mention 'skills' (DeepAgents concept). "
        "Without this, the classifier blocks 'What are skills?' as off-topic.\n\n"
        "Failing traces:\n"
        "  - 019cf6a3: 'What are skills?' — BLOCKED\n"
        "  - 019cf718: 'skills가 뭐야' — BLOCKED TWICE\n"
        "Fix: add 'Skills' to the DeepAgents terms in the ALWAYS ALLOW section."
    )


def test_guardrails_prompt_mentions_filesystemmiddleware_as_allowed():
    """The guardrails prompt must mention 'FilesystemMiddleware' in the ALWAYS ALLOW section.

    Production traces show the classifier blocks questions about FilesystemMiddleware
    (a DeepAgents middleware class) even in conversations with prior DeepAgents context.

    Failing traces:
      - 019cf8d9: "FilesystemMiddleware opowiedz mi o tym" — BLOCKED
      - 019cd791: "FilesystemMiddleware 的用法" — BLOCKED
    """
    allow_section = _get_always_allow_section()

    assert "filesystemmiddleware" in allow_section, (
        "The guardrails ALWAYS ALLOW section must mention 'FilesystemMiddleware' "
        "(a DeepAgents middleware class). Without this, the classifier blocks "
        "questions about it as off-topic.\n\n"
        "Failing traces:\n"
        "  - 019cf8d9: 'FilesystemMiddleware opowiedz mi o tym' — BLOCKED\n"
        "  - 019cd791: 'FilesystemMiddleware 的用法' — BLOCKED\n"
        "Fix: add 'FilesystemMiddleware' to DeepAgents terms in the ALWAYS ALLOW section."
    )


def test_guardrails_prompt_mentions_deepagents_in_always_allow():
    """The guardrails prompt must mention 'DeepAgents' in the ALWAYS ALLOW section.

    The classifier blocks questions about DeepAgents when the term appears alone
    without the space ("deep agents" vs "deepagents").

    Failing traces:
      - 019a4aff: "What are deep agents?" — BLOCKED
      - 019ab8e8/e9: "deepagents 的主要特征有那些" — BLOCKED TWICE
    """
    allow_section = _get_always_allow_section()

    has_deepagents = "deepagents" in allow_section or "deep agents" in allow_section
    assert has_deepagents, (
        "The guardrails ALWAYS ALLOW section must mention 'DeepAgents' or 'deep agents'. "
        "The classifier blocks 'deepagents' queries when it doesn't see this as LangChain.\n\n"
        "Failing traces:\n"
        "  - 019a4aff: 'What are deep agents?' — BLOCKED\n"
        "  - 019ab8e8: 'deepagents 的主要特征有那些' — BLOCKED TWICE\n"
        "Fix: ensure 'DeepAgents' appears in the ALWAYS ALLOW section."
    )


# ---------------------------------------------------------------------------
# LangSmith feature term tests
# ---------------------------------------------------------------------------


def test_guardrails_prompt_mentions_annotation_queues_as_allowed():
    """The guardrails prompt must mention 'Annotation Queues' in the ALWAYS ALLOW section.

    The classifier blocks "Annotation Queues" questions even though Annotation Queues
    is a core LangSmith feature for human labeling workflows.

    Failing traces:
      - 019cfa70-7003: "Annotation Queues 的数据表是什么样的？" — BLOCKED TWICE (same user!)
    """
    allow_section = _get_always_allow_section()

    assert "annotation queue" in allow_section, (
        "The guardrails ALWAYS ALLOW section must mention 'Annotation Queues' "
        "(a LangSmith feature). Without this, the classifier blocks "
        "LangSmith annotation queue questions as off-topic.\n\n"
        "Failing traces:\n"
        "  - 019cfa70-7003: 'Annotation Queues 的数据表是什么样的？' — BLOCKED TWICE\n"
        "Fix: add 'Annotation Queues' to LangSmith feature terms in the ALWAYS ALLOW section."
    )


def test_guardrails_prompt_mentions_langsmith_features():
    """The guardrails prompt must enumerate specific LangSmith features in ALWAYS ALLOW.

    The classifier needs to know what constitutes a LangSmith feature (Annotation Queues,
    Datasets, Experiments, etc.) to avoid blocking them.
    """
    allow_section = _get_always_allow_section()

    langsmith_features = ["annotation", "datasets", "experiments", "playground"]
    found = [f for f in langsmith_features if f in allow_section]

    assert len(found) >= 2, (
        f"The guardrails ALWAYS ALLOW section must mention at least 2 of these LangSmith "
        f"features: {langsmith_features}. Currently only found: {found}.\n\n"
        "This helps the classifier recognize LangSmith product terms as always-allowed topics."
    )


# ---------------------------------------------------------------------------
# Regression tests: existing allowed topics must not be removed
# ---------------------------------------------------------------------------


def test_guardrails_prompt_still_allows_core_langchain_topics():
    """Core LangChain topics must still appear in the ALWAYS ALLOW section (no regression)."""
    allow_section = _get_always_allow_section()

    core_topics = ["langchain", "langgraph", "langsmith", "mcp"]
    for topic in core_topics:
        assert topic in allow_section, (
            f"Core topic '{topic}' must still be present in the ALWAYS ALLOW section. "
            "The fix must not have accidentally removed essential allow-list entries."
        )


def test_guardrails_prompt_default_is_still_allow():
    """The default posture must still be ALLOW (no over-correction)."""
    has_allow_default = (
        "default is to allow" in PROMPT_LOWER
        or "your default is to allow" in PROMPT_LOWER
        or "when uncertain" in PROMPT_LOWER
    )
    assert has_allow_default, (
        "The guardrails prompt must still default to ALLOW. "
        "Verify the fix did not remove the 'YOUR DEFAULT IS TO ALLOW' language."
    )
