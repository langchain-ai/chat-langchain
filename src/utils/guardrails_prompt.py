"""Verify that served guardrails match the repository policy."""

import hashlib

from src.prompts.guardrails_prompts import guardrails_system_prompt


def verify_guardrails_prompt(content: str) -> str:
    """Return the SHA-256 hash or raise when Hub policy differs from local policy."""
    local_hash = hashlib.sha256(guardrails_system_prompt.encode()).hexdigest()
    hub_hash = hashlib.sha256(content.encode()).hexdigest()
    if hub_hash != local_hash:
        raise ValueError(
            f"Guardrails prompt drift: repository={local_hash}, hub={hub_hash}"
        )
    return hub_hash
