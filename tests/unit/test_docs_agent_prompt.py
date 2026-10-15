"""Tests for the docs agent prompt export."""

from pathlib import Path

from langchain_core.messages import SystemMessage

from scripts import push_docs_agent_prompt


def test_export_preserves_deployed_public_architecture(monkeypatch, capsys):
    exported_prompts = []

    class FakeClient:
        def push_prompt(self, repo_handle, **kwargs):
            exported_prompts.append(kwargs["object"])
            return "https://example.com/prompt"

    monkeypatch.setattr(push_docs_agent_prompt, "Client", FakeClient)
    monkeypatch.setattr(
        push_docs_agent_prompt, "load_dotenv", lambda *args, **kwargs: None
    )

    push_docs_agent_prompt.main()

    messages = exported_prompts[0].format_messages()
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    deployed_instructions = (
        Path(__file__).resolve().parents[2] / "instructions.md"
    ).read_text()
    about = (
        messages[0]
        .content.split("## About Chat LangChain (this assistant)", 1)[1]
        .split("## Your Mission", 1)[0]
    )
    deployed_about = deployed_instructions.split(
        "## About Chat LangChain (this assistant)", 1
    )[1].split("## Your Mission", 1)[0]
    assert about == deployed_about
    assert "Deep Agents" in about
    assert "Managed Deep Agent" in about
    assert "does not require a user-managed vector store" in about
    assert capsys.readouterr().out == "https://example.com/prompt\n"
