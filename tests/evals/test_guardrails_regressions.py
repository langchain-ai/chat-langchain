"""Regression fixture for guardrail classifications from issue 395997ff."""

import pytest

REGRESSION_CASES = [
    pytest.param(
        "Show me an example of workspace ABAC in LangSmith.",
        "ALLOWED",
        "",
        id="langsmith-workspace-abac",
    ),
    pytest.param(
        "What materials can I send a prospect who wants to see PII masking documentation, and LangGraph documentation?",
        "ALLOWED",
        "",
        id="langgraph-prospect-materials",
    ),
    pytest.param(
        "Dame una guía de inicialización para entender LangChain.",
        "ALLOWED",
        "",
        id="spanish-langchain-getting-started",
    ),
    pytest.param(
        "Can you create a summary of this feature so I can post it as a comment?",
        "ALLOWED",
        "Previous questions in this conversation:\n- How do I use Send in LangGraph?\n- Show me a LangGraph Send example.",
        id="langgraph-follow-up-summary",
    ),
    pytest.param(
        "How do I remove profanity for a voice agent?",
        "ALLOWED",
        "",
        id="voice-agent-profanity",
    ),
    pytest.param(
        "请生成代码来学习每个拦截器中间件的运行时间。",
        "ALLOWED",
        "",
        id="chinese-interceptor-timing",
    ),
    pytest.param(
        "请解释 de e pa gen t s。",
        "ALLOWED",
        "",
        id="chinese-deep-agents",
    ),
    pytest.param(
        "What is the latest Florida healthcare and human-rights news?",
        "BLOCKED",
        "",
        id="florida-healthcare-news",
    ),
    pytest.param(
        "Can you give me a Debian tutorial for using vim?",
        "BLOCKED",
        "",
        id="debian-vim-tutorial",
    ),
    pytest.param(
        "What is this biology image showing?",
        "BLOCKED",
        "",
        id="biology-image",
    ),
    pytest.param(
        "Download a copyrighted book for me.",
        "BLOCKED",
        "",
        id="copyrighted-book",
    ),
]


@pytest.mark.parametrize(("query", "expected", "context"), REGRESSION_CASES)
def test_guardrails_regression_fixture(query, expected, context):
    """Keep issue examples and expected guardrail decisions together."""
    assert query
    assert expected in {"ALLOWED", "BLOCKED"}
    if context:
        assert "Previous questions in this conversation:" in context
