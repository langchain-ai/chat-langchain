from langsmith import Client
from typing import Optional

from langsmith.utils import LangSmithNotFoundError

"""Default prompts."""

client = Client()


def _fetch_prompt(prompt_id: str) -> Optional[str]:
    """Safely fetch a prompt from LangSmith.

    If the prompt cannot be retrieved, return ``None`` and log a warning.
    """

    try:
        return client.pull_prompt(prompt_id).messages[0].prompt.template
    except Exception as exc:  # pragma: no cover - best effort fetch
        print(f"Warning: could not fetch prompt '{prompt_id}': {exc}")
        return None


INPUT_GUARDRAIL_SYSTEM_PROMPT = _fetch_prompt("margot-na/input_guardrail")
ROUTER_SYSTEM_PROMPT = _fetch_prompt("margot-na/router")
GENERATE_QUERIES_SYSTEM_PROMPT = _fetch_prompt(
    "margot-na/generate-queries"
)
MORE_INFO_SYSTEM_PROMPT = _fetch_prompt("margot-na/more_info")
RESEARCH_PLAN_SYSTEM_PROMPT = _fetch_prompt("margot-na/researcher")
GENERAL_SYSTEM_PROMPT = _fetch_prompt("margot-na/irrelevant_response")
RESPONSE_SYSTEM_PROMPT = _fetch_prompt("margot-na/synthesizer")
