"""Tests for the create_agent() parameter guardrail in the docs agent prompt.

Root cause: The agent was recommending `prompt=` (old create_react_agent parameter)
and `system_prompt=<ChatPromptTemplate>` (wrong type) for create_agent(), causing
TypeErrors in user code.

Fix: Added an explicit API Parameter Note section to docs_agent_prompt that
clarifies the correct `system_prompt=` parameter and its accepted types.

Failing traces:
  - 019d1415: Agent says create_agent(llm, tools=tools, prompt=prompt) is valid
  - 019d1416: User gets TypeError from following that advice
  - 019d1419: Agent says system_prompt=ChatPromptTemplate is valid, user gets another error
  - 019d1411: Agent tells Korean user to pass prompt as positional arg
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.prompts.docs_agent_prompt import docs_agent_prompt

PROMPT_LOWER = docs_agent_prompt.lower()


def test_prompt_warns_against_prompt_kwarg():
    """Prompt must explicitly flag that 'prompt=' is NOT valid for create_agent().

    This prevents the agent from recommending the old create_react_agent() kwarg.
    """
    assert "prompt=" in PROMPT_LOWER, (
        "The prompt must mention 'prompt=' in order to warn against its use "
        "with create_agent(). Without this, the agent will keep recommending "
        "the deprecated create_react_agent() parameter name."
    )


def test_prompt_requires_system_prompt_kwarg():
    """Prompt must tell the agent that create_agent() uses system_prompt=.

    The correct API is create_agent(llm, tools=tools, system_prompt=<str or SystemMessage>).
    """
    assert "system_prompt=" in PROMPT_LOWER, (
        "The prompt must mention 'system_prompt=' to guide the agent toward the "
        "correct create_agent() keyword argument."
    )


def test_prompt_warns_against_chatprompttemplate():
    """Prompt must warn that ChatPromptTemplate is not accepted by system_prompt=.

    The agent was previously saying system_prompt=<ChatPromptTemplate> works,
    which causes a TypeError at runtime.
    """
    assert "chatprompttemplate" in PROMPT_LOWER, (
        "The prompt must mention 'ChatPromptTemplate' to explicitly warn that "
        "it is NOT an accepted type for create_agent()'s system_prompt= parameter."
    )


def test_prompt_clarifies_system_prompt_accepts_string_or_systemmessage():
    """Prompt must state that system_prompt= accepts a str or SystemMessage."""
    has_str = "str" in PROMPT_LOWER
    has_systemmessage = "systemmessage" in PROMPT_LOWER
    assert has_str and has_systemmessage, (
        "The prompt must clarify that system_prompt= accepts a str or SystemMessage. "
        f"Found str={has_str}, SystemMessage={has_systemmessage}."
    )


def test_prompt_references_create_agent():
    """Prompt must reference create_agent() by name to be specific about this API."""
    assert "create_agent" in PROMPT_LOWER, (
        "The prompt must explicitly reference 'create_agent' so the guidance "
        "is unambiguous and not confused with other agent constructors."
    )


def test_prompt_distinguishes_from_create_react_agent():
    """Prompt must mention create_react_agent to make the old/new API distinction clear."""
    assert "create_react_agent" in PROMPT_LOWER, (
        "The prompt must mention 'create_react_agent' to explain that prompt= "
        "belongs to the old API, not create_agent(). This prevents the agent "
        "from confusing migration docs that show both APIs side-by-side."
    )
