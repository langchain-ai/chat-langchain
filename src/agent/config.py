"""Shared configuration for all agents."""

import logging
import os
import re
from dataclasses import dataclass

import dotenv
from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable, RunnableLambda

from src.middleware.model_experiment_middleware import (
    ARM_BASE,
    ARM_CONTROL,
    ARM_TREATMENT,
    ModelArmPinMiddleware,
    ModelChain,
    ModelExperiment,
    ModelExperimentFallbackMiddleware,
)
from src.middleware.retry_middleware import (
    RETRYABLE_FINISH_REASONS,
    MalformedResponseError,
    ModelRetryMiddleware,
)
from src.middleware.tool_retry_middleware import ToolRetryMiddleware

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

# =============================================================================
# Model Registry
# =============================================================================


@dataclass
class ModelConfig:
    """Configuration for a supported chat model."""

    id: str  # e.g., "google_genai:gemini-3.5-flash-lite"
    name: str  # Display name, e.g., "Gemini 3.5 Flash Lite"
    provider: str  # e.g., "google", "openai", "baseten"
    api_key_env: str  # Environment variable for API key
    description: str | None = None


# Backend-supported models.
MODELS: dict[str, ModelConfig] = {
    # Anthropic
    "claude-haiku-4.5": ModelConfig(
        id="anthropic:claude-haiku-4-5-20251001",
        name="Claude Haiku 4.5",
        provider="anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        description="Fast and cheap Anthropic model",
    ),
    # OpenAI
    "gpt-5.4-nano": ModelConfig(
        id="openai:gpt-5.4-nano",
        name="GPT-5.4 Nano",
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        description="Cheapest GPT-5.4-class model for simple high-volume tasks",
    ),
    # Google
    "gemini-3.5-flash-lite": ModelConfig(
        id="google_genai:gemini-3.5-flash-lite",
        name="Gemini 3.5 Flash Lite",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
        description="Fastest, most cost-effective Gemini",
    ),
}

# Default models for different use cases
DEFAULT_MODEL = MODELS["gemini-3.5-flash-lite"]
GUARDRAILS_MODEL = MODELS["gpt-5.4-nano"]

# Fallback chain (in order of preference)
FALLBACK_MODELS = [
    MODELS["gpt-5.4-nano"],
    MODELS["claude-haiku-4.5"],
]

# =============================================================================
# Post-trained model experiment
# =============================================================================
# A Fireworks serverless LoRA serves FT_MODEL_TRAFFIC_PERCENT of new threads,
# the base model it was tuned from (FT_BASE_MODEL_ID, also on Fireworks)
# serves FT_BASE_MODEL_TRAFFIC_PERCENT, and DEFAULT_MODEL serves the rest.
# Both percents default to 33, so the three arms split new threads ~1/3 each
# (33/33/34). Leave FT_BASE_MODEL_ID unset for a two-arm split; the base arm
# only runs while FT_MODEL_ID is set. Both come from the deployment environment
# (GitHub repository variables in the deploy workflow), so changing them is
# a redeploy, not a code change. Load the adapter with
#   firectl model load-lora <model-id> --serverless
# and set FT_MODEL_ID to its model resource name, without a `#deployment`
# suffix: "accounts/<account-id>/models/<model-id>". Needs FIREWORKS_API_KEY.
#
# FT_MODEL_ID unset or empty keeps every thread on DEFAULT_MODEL, including
# threads already pinned to treatment: unsetting it is the kill switch.
# FT_MODEL_TRAFFIC_PERCENT=0 instead stops new enrollment and keeps pinned
# threads as they are. Change FT_MODEL_EXPERIMENT only for a new experiment:
# it salts the thread hash and labels every model-call run in LangSmith.
#
# Swapping FT_MODEL_ID to a new LoRA moves already-open treatment threads onto
# the new model too: the pinned arm doesn't record which model it came from.
# Change FT_MODEL_EXPERIMENT as well for a clean break; old treatment threads
# still run the new FT_MODEL_ID, and their later turns carry the new label
# (only earlier turns keep the old one). See
# src/middleware/model_experiment_middleware.py.
FT_MODEL_EXPERIMENT = "clc-ft-model-v1"
_DEFAULT_FT_MODEL_TRAFFIC_PERCENT = 33

_FIREWORKS_MODEL_ID = re.compile(r"accounts/[^/#\s]+/models/[^/#\s]+")


def _ft_model_id_from_env(env: str = "FT_MODEL_ID") -> str | None:
    model_id = (os.getenv(env) or "").strip() or None
    if model_id is not None and not _FIREWORKS_MODEL_ID.fullmatch(model_id):
        raise ValueError(
            f"{env} must look like accounts/<account-id>/models/<model-id> "
            f"(a serverless model, no #deployment suffix); got {model_id!r}"
        )
    return model_id


def _ft_traffic_percent_from_env(env: str = "FT_MODEL_TRAFFIC_PERCENT") -> int:
    raw = (os.getenv(env) or "").strip()
    if not raw:
        return _DEFAULT_FT_MODEL_TRAFFIC_PERCENT
    try:
        percent = int(raw)
    except ValueError:
        percent = -1
    if not 0 <= percent <= 100:
        raise ValueError(f"{env} must be a whole number 0-100; got {raw!r}")
    return percent


FT_MODEL_ID = _ft_model_id_from_env()
FT_MODEL_TRAFFIC_PERCENT = _ft_traffic_percent_from_env()
FT_BASE_MODEL_ID = _ft_model_id_from_env("FT_BASE_MODEL_ID")

# reasoning_effort for both Fireworks arms. "high" matches the offline evals;
# leaving it unset on Fireworks gives GLM far more thinking (2-5x the tokens,
# much slower). GLM-5.3 is thinking-only, so "none" is rejected with a 400.
# Fireworks accepts low, high and max for GLM (no medium).
_FT_REASONING_EFFORTS = ("low", "high", "max")


def _ft_reasoning_effort_from_env() -> str | None:
    raw = (os.getenv("FT_REASONING_EFFORT") or "high").strip().lower()
    if raw in ("default", "unset"):
        return None
    if raw not in _FT_REASONING_EFFORTS:
        raise ValueError(
            f"FT_REASONING_EFFORT must be one of low, high, max or default; got {raw!r}"
        )
    return raw


FT_REASONING_EFFORT = _ft_reasoning_effort_from_env()
FT_BASE_MODEL_TRAFFIC_PERCENT = _ft_traffic_percent_from_env(
    "FT_BASE_MODEL_TRAFFIC_PERCENT"
)

# =============================================================================
# API Key Setup
# =============================================================================

API_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "FIREWORKS_API_KEY",
]

for key in API_KEYS:
    if value := os.getenv(key):
        os.environ[key] = value.strip()
        logger.info(f"{key} configured")


# =============================================================================
# Model Initialization
# =============================================================================

# Retry configuration
MAX_RETRIES = int(os.getenv("MODEL_MAX_RETRIES", "2"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "16384"))
MODEL_REQUEST_TIMEOUT = float(os.getenv("MODEL_REQUEST_TIMEOUT", "60"))


def _experiment_metadata(arm: str, role: str) -> dict[str, str]:
    """LangSmith metadata naming the arm and role a model call served."""
    metadata = {"model_role": role}
    if FT_MODEL_ID is not None:
        metadata |= {"model_experiment": FT_MODEL_EXPERIMENT, "model_arm": arm}
    return metadata


# Primary model (control arm). Public callers cannot switch this at runtime.
default_model = init_chat_model(
    model=DEFAULT_MODEL.id,
    max_tokens=MODEL_MAX_TOKENS,
    timeout=MODEL_REQUEST_TIMEOUT,
    metadata=_experiment_metadata(ARM_CONTROL, "primary"),
)
logger.info(f"Default model: {DEFAULT_MODEL.name} ({DEFAULT_MODEL.id})")


def _raise_for_retryable_finish_reason(response: object) -> object:
    metadata = getattr(response, "response_metadata", None) or {}
    finish_reason = metadata.get("finish_reason", "")
    if finish_reason in RETRYABLE_FINISH_REASONS:
        raise MalformedResponseError(f"Model returned {finish_reason}")
    return response


def _init_retrying_model(model: str) -> Runnable:
    return (
        init_chat_model(
            model=model, max_tokens=MODEL_MAX_TOKENS, timeout=MODEL_REQUEST_TIMEOUT
        )
        | RunnableLambda(_raise_for_retryable_finish_reason)
    ).with_retry(stop_after_attempt=MAX_RETRIES + 1)


def init_retry_fallback_model(model: str) -> Runnable:
    """Initialize a model runnable with the shared retry and fallback policy."""
    primary_model = _init_retrying_model(model)
    fallback_models = [
        _init_retrying_model(fallback.id) for fallback in FALLBACK_MODELS
    ]
    return primary_model.with_fallbacks(fallback_models)


summarization_model = init_retry_fallback_model(DEFAULT_MODEL.id)

# =============================================================================
# Middleware
# =============================================================================

model_retry_middleware = ModelRetryMiddleware(max_retries=MAX_RETRIES)
tool_retry_middleware = ToolRetryMiddleware(max_attempts=3)


def _experiment_model(model: str, *, arm: str, role: str, **kwargs: object):
    """Init a chat model whose LangSmith runs say which arm and role they served."""
    return init_chat_model(
        model=model,
        max_tokens=MODEL_MAX_TOKENS,
        timeout=MODEL_REQUEST_TIMEOUT,
        metadata=_experiment_metadata(arm, role),
        **kwargs,
    )


def _fallback_chain(arm: str, models: list[ModelConfig]) -> tuple:
    return tuple(_experiment_model(m.id, arm=arm, role="fallback") for m in models)


def _fireworks_chain(model_id: str, arm: str) -> ModelChain:
    if not os.getenv("FIREWORKS_API_KEY"):
        raise RuntimeError(
            "FT_MODEL_ID is set but FIREWORKS_API_KEY is missing; add the "
            "FIREWORKS_API_KEY secret or unset FT_MODEL_ID."
        )
    return ModelChain(
        primary=_experiment_model(
            model_id,
            arm=arm,
            role="primary",
            model_provider="fireworks",
            **(
                {"reasoning_effort": FT_REASONING_EFFORT} if FT_REASONING_EFFORT else {}
            ),
        ),
        # Fireworks arms fall back to the control model first, then the usual chain.
        fallbacks=_fallback_chain(arm, [DEFAULT_MODEL, *FALLBACK_MODELS]),
    )


def _treatment_chain() -> ModelChain | None:
    if FT_MODEL_ID is None:
        return None
    return _fireworks_chain(FT_MODEL_ID, ARM_TREATMENT)


def _base_chain() -> ModelChain | None:
    # The base arm is only a comparison for the post-trained model, so the
    # FT_MODEL_ID kill switch turns it off too.
    if FT_MODEL_ID is None or FT_BASE_MODEL_ID is None:
        return None
    return _fireworks_chain(FT_BASE_MODEL_ID, ARM_BASE)


model_experiment = ModelExperiment(
    name=FT_MODEL_EXPERIMENT,
    control=ModelChain(
        primary=default_model,
        fallbacks=_fallback_chain(ARM_CONTROL, FALLBACK_MODELS),
    ),
    treatment=_treatment_chain(),
    treatment_percent=FT_MODEL_TRAFFIC_PERCENT,
    base=_base_chain(),
    base_percent=FT_BASE_MODEL_TRAFFIC_PERCENT,
)
# Goes first in the middleware list so the arm is pinned on every first turn.
model_arm_pin_middleware = ModelArmPinMiddleware(model_experiment)
model_fallback_middleware = ModelExperimentFallbackMiddleware(model_experiment)
logger.info(f"Fallback chain: {' -> '.join(m.name for m in FALLBACK_MODELS)}")
if model_experiment.enabled:
    logger.info(
        f"Model experiment {FT_MODEL_EXPERIMENT}: {FT_MODEL_TRAFFIC_PERCENT}% of "
        f"new threads -> {FT_MODEL_ID}"
    )
if model_experiment.base_enabled:
    logger.info(
        f"Model experiment {FT_MODEL_EXPERIMENT}: {FT_BASE_MODEL_TRAFFIC_PERCENT}% "
        f"of new threads -> base {FT_BASE_MODEL_ID}"
    )

# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Models
    "MODELS",
    "DEFAULT_MODEL",
    "GUARDRAILS_MODEL",
    "FALLBACK_MODELS",
    "ModelConfig",
    # Models
    "default_model",
    "init_retry_fallback_model",
    "summarization_model",
    # Middleware
    "model_retry_middleware",
    "tool_retry_middleware",
    "model_fallback_middleware",
    "model_arm_pin_middleware",
    "model_experiment",
    # Config
    "FT_MODEL_ID",
    "FT_MODEL_TRAFFIC_PERCENT",
    "FT_BASE_MODEL_ID",
    "FT_BASE_MODEL_TRAFFIC_PERCENT",
    "FT_REASONING_EFFORT",
    "FT_MODEL_EXPERIMENT",
    "MAX_RETRIES",
    "logger",
]
