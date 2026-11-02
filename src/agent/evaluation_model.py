"""Opt-in model selection for isolated evaluation deployments."""

import os

from langchain_openai import ChatOpenAI


def get_evaluation_model() -> ChatOpenAI | None:
    """Build the Kimi evaluation model without changing other model clients."""
    selection = os.getenv("CLC_EVAL_MODEL", "").strip()
    if not selection:
        return None
    if selection != "kimi-k3":
        raise ValueError("CLC_EVAL_MODEL must be unset or 'kimi-k3'")
    key = os.getenv("FIREWORKS_API_KEY", "").strip()
    if not key:
        raise ValueError("FIREWORKS_API_KEY is required for CLC_EVAL_MODEL=kimi-k3")
    return ChatOpenAI(
        model="accounts/fireworks/models/kimi-k3",
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=key,
        use_responses_api=False,
    )
