"""Model selection must isolate Fireworks from production provider settings."""

import os
import subprocess
import sys

import pytest
from pydantic import SecretStr

from src.agent.evaluation_model import get_evaluation_model


def test_unset_evaluation_model_preserves_default(monkeypatch):
    monkeypatch.delenv("CLC_EVAL_MODEL", raising=False)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    assert get_evaluation_model() is None


def test_unknown_evaluation_model_fails(monkeypatch):
    monkeypatch.setenv("CLC_EVAL_MODEL", "unknown")
    with pytest.raises(ValueError, match="CLC_EVAL_MODEL"):
        get_evaluation_model()


def test_missing_fireworks_key_fails(monkeypatch):
    monkeypatch.setenv("CLC_EVAL_MODEL", "kimi-k3")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
        get_evaluation_model()


def test_kimi_uses_its_own_provider_settings(monkeypatch):
    monkeypatch.setenv("CLC_EVAL_MODEL", "kimi-k3")
    monkeypatch.setenv("FIREWORKS_API_KEY", "offline-test-placeholder")
    monkeypatch.setenv("OPENAI_API_KEY", "other-offline-placeholder")
    monkeypatch.setenv(
        "OPENAI_BASE_URL", "https://gateway.smith.langchain.com/openai/v1"
    )
    model = get_evaluation_model()
    assert model.model_name == "accounts/fireworks/models/kimi-k3"
    assert model.openai_api_base == "https://api.fireworks.ai/inference/v1"
    assert model.use_responses_api is False
    assert model.openai_api_key == SecretStr("offline-test-placeholder")
    # Selection must not rewrite the environment used by guardrail clients.
    assert os.environ["OPENAI_API_KEY"] == "other-offline-placeholder"
    assert (
        os.environ["OPENAI_BASE_URL"] == "https://gateway.smith.langchain.com/openai/v1"
    )


@pytest.mark.parametrize("selection", ["", "kimi-k3"])
def test_agent_wiring_preserves_auxiliary_models(selection):
    environment = {
        **os.environ,
        "USE_LOCAL_PROMPTS": "1",
        "GOOGLE_API_KEY": "offline-test-placeholder",
        "ANTHROPIC_API_KEY": "offline-test-placeholder",
        "OPENAI_API_KEY": "offline-test-placeholder",
        "FIREWORKS_API_KEY": "offline-test-placeholder",
        "CLC_EVAL_MODEL": selection,
        "CLC_DISABLE_GUARDRAIL_SAMPLE_COLLECTION": "true",
    }
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import agent
from langchain.agents.middleware import ModelFallbackMiddleware
from src.middleware.guardrails_middleware import GuardrailsMiddleware
is_kimi = os.environ['CLC_EVAL_MODEL'] == 'kimi-k3'
model = agent.agent.config['model']
if is_kimi:
    assert model.model_name == 'accounts/fireworks/models/kimi-k3'
else:
    assert model == 'google_genai:gemini-3.5-flash-lite'
assert any(isinstance(m, ModelFallbackMiddleware) for m in agent.docs_agent_middleware) != is_kimi
guard = next(m for m in agent.docs_agent_middleware if isinstance(m, GuardrailsMiddleware))
assert guard.classifier_llms[0][0] == 'openai:gpt-5.4-nano'
assert guard.collect_samples is False
""",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
