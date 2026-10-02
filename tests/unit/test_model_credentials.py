import importlib
import sys

from langchain_core.runnables import RunnableLambda


class FakeModel(RunnableLambda):
    def __init__(self, error=None):
        self.error = error
        super().__init__(self._invoke)

    def _invoke(self, value, config=None, **kwargs):
        if self.error:
            raise self.error
        return value


class FakeModels:
    def __init__(self, error=None):
        self.error = error
        self.retrieved = []

    def retrieve(self, model_id):
        self.retrieved.append(model_id)
        if self.error:
            raise self.error


class FakeOpenAI:
    def __init__(self, error=None):
        self.models = FakeModels(error)


def _load_config(monkeypatch, models, openai_client):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setattr(
        "langchain.chat_models.init_chat_model",
        lambda model: models.setdefault(model, FakeModel()),
    )
    monkeypatch.setattr("openai.OpenAI", lambda: openai_client)
    sys.modules.pop("src.agent.config", None)
    return importlib.import_module("src.agent.config")


def test_startup_credential_check_uses_openai_model_retrieval(monkeypatch):
    models = {}
    openai_client = FakeOpenAI()
    config = _load_config(monkeypatch, models, openai_client)

    assert config.provider_authentication == {
        "anthropic": True,
        "openai": True,
        "google": True,
    }
    assert openai_client.models.retrieved == ["gpt-5.4-nano"]


def test_startup_credential_check_is_nonfatal(monkeypatch, caplog):
    models = {}
    openai_client = FakeOpenAI(RuntimeError("invalid api key"))
    config = _load_config(monkeypatch, models, openai_client)

    assert config.provider_authentication["openai"] is False
    assert "Provider authentication check failed for openai" in caplog.text


def test_output_limit_error_counts_as_authenticated(monkeypatch):
    models = {
        "anthropic:claude-haiku-4-5-20251001": FakeModel(
            RuntimeError(
                "Could not finish the message because max_tokens or model output limit was reached."
            )
        )
    }
    openai_client = FakeOpenAI()
    config = _load_config(monkeypatch, models, openai_client)

    assert config.provider_authentication["anthropic"] is True
