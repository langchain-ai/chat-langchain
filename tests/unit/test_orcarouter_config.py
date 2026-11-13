"""Tests for the OrcaRouter model registry entry and gateway-aware init."""

from src.agent import config


def test_orcarouter_is_a_registered_model():
    """OrcaRouter should be a first-class model registry entry."""
    assert "orcarouter" in config.MODELS
    orcarouter = config.MODELS["orcarouter"]
    assert orcarouter.provider == "orcarouter"
    assert orcarouter.api_key_env == "ORCAROUTER_API_KEY"
    assert orcarouter.id == "openai:orcarouter/auto"


def test_init_configured_model_applies_gateway_defaults(monkeypatch):
    """OrcaRouter models should route to the gateway base URL."""
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test-key")
    llm = config.init_configured_model(config.MODELS["orcarouter"].id, temperature=0)
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "orcarouter/auto"
    assert llm.openai_api_base == "https://api.orcarouter.ai/v1"


def test_init_configured_model_uses_env_base_url_override(monkeypatch):
    """ORCAROUTER_BASE_URL should override the default gateway endpoint."""
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test-key")
    monkeypatch.setenv("ORCAROUTER_BASE_URL", "https://proxy.example.com/v1")
    llm = config.init_configured_model(config.MODELS["orcarouter"].id)
    assert llm.openai_api_base == "https://proxy.example.com/v1"


def test_init_configured_model_injects_gateway_key(monkeypatch):
    """ORCAROUTER_API_KEY should be forwarded to the OpenAI-compatible client."""
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = config.init_configured_model(config.MODELS["orcarouter"].id)
    assert llm.openai_api_key.get_secret_value() == "sk-orca-test-key"


def test_init_configured_model_does_not_affect_other_providers(monkeypatch):
    """Non-OrcaRouter models should keep their native provider wiring."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key")
    monkeypatch.setenv("ORCAROUTER_API_KEY", "sk-orca-test-key")
    llm = config.init_configured_model(config.MODELS["gpt-5.4-nano"].id)
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "gpt-5.4-nano"
    assert llm.openai_api_base is None
    assert llm.openai_api_key.get_secret_value() == "sk-openai-test-key"
