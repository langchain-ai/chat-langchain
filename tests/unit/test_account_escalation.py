"""Tests for account issue escalation guidance and ticket creation."""

from unittest.mock import Mock, patch

import pytest

from src.prompts.docs_agent_prompt import docs_agent_prompt
from src.tools.pylon_tools import create_support_ticket

ACCOUNT_REQUESTS = [
    ("our account was suspended, need human support", "suspen"),
    ("help with failed payment", "payment"),
    ("can you raise a ticket with support", "ticket"),
]


@pytest.mark.parametrize("user_request, intent", ACCOUNT_REQUESTS)
def test_account_requests_use_escalation_branch(user_request, intent):
    prompt = docs_agent_prompt.lower()

    assert intent in user_request.lower()
    assert intent in prompt
    assert "cannot change account state" in prompt
    assert "create_support_ticket" in prompt
    assert "quote the returned ticket reference" in prompt


def test_documentation_question_is_not_account_escalation():
    request = "how do I stream tokens from a LangGraph node"
    account_terms = ("account", "billing", "payment", "suspension", "outage")

    assert not any(term in request.lower() for term in account_terms)
    assert "documentation and usage questions" in docs_agent_prompt.lower()


@patch("src.tools.pylon_tools.requests.post")
@patch("src.tools.pylon_tools._get_api_key", return_value="fake-key")
def test_create_support_ticket_returns_id_with_runtime_context(mock_api_key, mock_post):
    response = Mock()
    response.json.return_value = {"data": {"id": "ticket-123"}}
    mock_post.return_value = response

    result = create_support_ticket.invoke(
        {"subject": "Account suspended", "body": "Please help."},
        config={"configurable": {"thread_id": "thread-456", "user_id": "user-789"}},
    )

    assert result == "ticket-123"
    assert mock_post.call_args.args[0] == "https://api.usepylon.com/v1/issues"
    payload = mock_post.call_args.kwargs["json"]
    assert payload["body_html"] == "Please help."
    assert payload["metadata"] == {
        "thread_id": "thread-456",
        "user_id": "user-789",
    }
