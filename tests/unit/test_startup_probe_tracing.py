import os
from contextlib import contextmanager

os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")

from src.agent import config


def test_invoke_startup_probe_disables_tracing(monkeypatch):
    tracing_states: list[bool] = []

    @contextmanager
    def fake_tracing_context(*, enabled: bool):
        tracing_states.append(enabled)
        yield

    class FakeModel:
        def invoke(self, prompt, **kwargs):
            return prompt, kwargs

    monkeypatch.setattr(config, "tracing_context", fake_tracing_context)

    result = config.invoke_startup_probe(FakeModel(), "Reply with OK", timeout=1)

    assert result == ("Reply with OK", {"timeout": 1})
    assert tracing_states == [False]
