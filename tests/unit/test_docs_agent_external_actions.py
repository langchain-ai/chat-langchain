import pytest

from src.prompts.docs_agent_prompt import docs_agent_prompt


@pytest.mark.parametrize(
    ("user_request", "requires_inline_delivery"),
    [
        (
            "Give me a two-line summary of what LangGraph is, and save or export it as notes.md so I can download it",
            True,
        ),
        (
            "Summarize how LangGraph checkpoints work, save the summary as /tmp/langgraph-checkpoints.md, and tell me when the file is ready",
            True,
        ),
        (
            "Run python -c 'import langgraph; print(langgraph.__version__)' in your environment to verify the installed LangGraph version",
            False,
        ),
    ],
)
def test_external_action_requests_are_covered_by_prompt(
    user_request, requires_inline_delivery
):
    """Prompt requires refusal and inline delivery for external actions."""
    prompt = docs_agent_prompt.lower()
    request_lower = user_request.lower()

    assert any(
        term in request_lower
        for term in ("save", "export", "download", "run", "verify")
    )
    assert "server the user cannot see" in prompt
    assert "no access to the user's machine, filesystem, or shell" in prompt
    assert "never claim to have saved, written, exported, downloaded, or delivered a file" in prompt
    assert "never present a `write_file` tool result as a file the user can open or retrieve" in prompt
    assert "never state or imply that you executed a command or verified an installed package version" in prompt
    assert "you cannot run code" in prompt
    assert "say plainly in the same answer that you cannot perform the action" in prompt
    assert "never silently drop the action half of the request" in prompt
    if requires_inline_delivery:
        assert "answer the informational part inline in a fenced block the user can copy" in prompt
