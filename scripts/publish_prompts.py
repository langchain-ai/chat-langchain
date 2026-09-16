"""Publish all production prompts and fail closed on API errors."""

from __future__ import annotations

import re

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from src.prompts.docs_agent_prompt import docs_agent_prompt
from src.prompts.guardrails_prompts import guardrails_system_prompt

_PROMPTS = (
    (
        "src/prompts/docs_agent_prompt.py",
        "langchain-ai/public-chat-langchain-test",
        docs_agent_prompt,
    ),
    (
        "src/prompts/guardrails_prompts.py",
        "langchain-ai/public-chat-langchain-guardrails-test",
        guardrails_system_prompt,
    ),
)


def _status_code(exc: Exception) -> str:
    """Extract an HTTP status code when the client exposes one."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status is not None:
        return str(status)
    match = re.search(r"\b([45]\d{2})\b", str(exc))
    return match.group(1) if match else "unknown"


def main() -> None:
    """Publish production prompts or exit non-zero."""
    load_dotenv(".env", override=True)
    client = Client()
    for source_path, hub_handle, prompt_text in _PROMPTS:
        prompt = ChatPromptTemplate.from_messages(
            [SystemMessage(content=prompt_text)]
        )
        try:
            client.push_prompt(hub_handle, object=prompt)
        except Exception as exc:
            raise SystemExit(
                f"Prompt Hub publish failed: source={source_path} "
                f"handle={hub_handle} status={_status_code(exc)}: {exc}"
            ) from exc


if __name__ == "__main__":
    main()
