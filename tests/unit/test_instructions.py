from pathlib import Path
from unittest.mock import MagicMock

from scripts import push_docs_agent_prompt


def test_yes_no_answers_require_retrieved_evidence():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "For a yes/no question, first check whether a retrieved" in instructions
    assert "the documentation does not state the claim" in instructions
    assert "require a quoted source from the retrieved content" in instructions


def test_cross_conversation_memory_limit_is_explicit():
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()

    assert "You cannot remember information across conversations" in instructions
    assert "each new chat starts fresh" in instructions
    assert (
        "If a user asks you to remember something for future conversations"
        in instructions
    )
    assert (
        "tell them plainly you cannot retain it beyond this conversation"
        in instructions
    )
    assert "then offer to help with their LangChain question" in instructions


def test_docs_prompt_publisher_uses_current_instructions(monkeypatch):
    instructions = (Path(__file__).parents[2] / "instructions.md").read_text()
    client = MagicMock()
    monkeypatch.setattr(push_docs_agent_prompt, "Client", lambda: client)
    monkeypatch.setattr(push_docs_agent_prompt, "load_dotenv", MagicMock())

    push_docs_agent_prompt.main()

    client.push_prompt.assert_called_once()
    assert client.push_prompt.call_args.args == (
        "langchain-ai/public-chat-langchain-test",
    )
    prompt = client.push_prompt.call_args.kwargs["object"]
    messages = prompt.format_messages()
    assert len(messages) == 1
    assert messages[0].type == "system"
    assert messages[0].content == instructions
