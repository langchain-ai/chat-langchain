import importlib
import sys

import pytest
from langchain_core.runnables import RunnableLambda


class FakeModel(RunnableLambda):
    def __init__(self, model_id, error=None):
        self.model_id = model_id
        self.error = error
        super().__init__(self._invoke)

    def _invoke(self, value, config=None, **kwargs):
        if self.error:
            raise self.error
        return value


def _load_config(monkeypatch, models):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model",
        lambda model: models.setdefault(model, FakeModel(model)),
    )
    sys.modules.pop("src.agent.config", None)
    return importlib.import_module("src.agent.config")


def test_startup_credential_check_raises_on_rejected_credential(monkeypatch):
    config = _load_config(monkeypatch, {})
    config.init_chat_model = lambda model: FakeModel(
        model, error=RuntimeError("API_KEY_INVALID")
    )

    with pytest.raises(RuntimeError, match="Credential validation failed"):
        config.validate_configured_model_credentials()


def test_startup_credential_check_passes_valid_credentials(monkeypatch):
    models = {}
    config = _load_config(monkeypatch, models)

    config.validate_configured_model_credentials()

    assert set(models) == {
        "google_genai:gemini-3.5-flash-lite",
        "openai:gpt-5.4-nano",
        "anthropic:claude-haiku-4-5-20251001",
    }
