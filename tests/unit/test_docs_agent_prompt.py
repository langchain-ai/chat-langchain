"""Tests for publishing the docs agent's public self-description."""

from pathlib import Path

from langchain_core.messages import SystemMessage

from scripts import push_docs_agent_prompt


def test_published_prompt_preserves_runtime_public_facts(monkeypatch, capsys):
    published = []

    class FakeClient:
        def push_prompt(self, name, *, object):
            published.append((name, object.invoke({}).messages))
            return "https://smith.langchain.com/prompts/test"

    monkeypatch.setattr(push_docs_agent_prompt, "Client", FakeClient)
    monkeypatch.setattr(
        push_docs_agent_prompt, "load_dotenv", lambda *args, **kwargs: None
    )

    push_docs_agent_prompt.main()

    assert len(published) == 1
    name, messages = published[0]
    assert name == "langchain-ai/public-chat-langchain-test"
    assert len(messages) == 1
    assert isinstance(messages[0], SystemMessage)
    assert capsys.readouterr().out == "https://smith.langchain.com/prompts/test\n"

    runtime_prompt = (
        Path(__file__).resolve().parents[2] / "instructions.md"
    ).read_text()
    descriptions = []
    for prompt in (runtime_prompt, messages[0].content):
        public_facts = prompt.split("## About this assistant (public facts)\n", 1)[1]
        public_facts = public_facts.split("\n## Your Mission", 1)[0]
        assert "https://github.com/langchain-ai/chat-langchain" in public_facts
        assert "no vector store or embeddings index of your own" in public_facts
        descriptions.append(public_facts)

    assert descriptions[0] == descriptions[1]
