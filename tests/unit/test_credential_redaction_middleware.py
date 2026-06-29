"""Tests for input-side credential redaction middleware."""

from langchain_core.messages import HumanMessage

from src.middleware.credential_redaction_middleware import (
    CredentialRedactionMiddleware,
)


def test_before_agent_redacts_openai_style_key():
    """sk-* keys in HumanMessage content are rewritten to a placeholder."""
    raw_key = "sk-1234567890abcdef1234567890abcdef"
    msg = HumanMessage(
        content=f'model = ChatTongyi(dashscope_api_key="{raw_key}")'
    )

    result = CredentialRedactionMiddleware().before_agent(
        {"messages": [msg]}, runtime=None
    )

    assert result is not None
    updated = result["messages"]
    assert len(updated) == 1
    redacted = updated[0]
    assert isinstance(redacted, HumanMessage)
    assert "sk-***REDACTED***" in redacted.content
    assert raw_key not in redacted.content
    assert redacted.additional_kwargs["credentials_redacted"] is True
    assert "sk-[A-Za-z0-9]{32,}" in redacted.additional_kwargs["patterns_matched"]


def test_before_agent_returns_none_for_clean_message():
    """Messages with no credential patterns leave state untouched."""
    msg = HumanMessage(content="How do I use create_agent in LangChain?")

    result = CredentialRedactionMiddleware().before_agent(
        {"messages": [msg]}, runtime=None
    )

    assert result is None
