"""Regression checks for Python and JavaScript SDK separation."""

import re
from pathlib import Path

import pytest
from langsmith import testing as t

from src.prompts.docs_agent_prompt import docs_agent_prompt

SDK_RULE = "Never mix SDKs inside one code block."
LANGUAGE_SEARCH_RULE = "always prefix the query with `python` or `javascript`"
JAVASCRIPT_METHODS = (
    "fromMessages",
    "fromTemplate",
    "addNode",
    "addEdge",
    "addConditionalEdges",
    "withConfig",
    "bindTools",
    "withStructuredOutput",
)


def _python_code_blocks(response: str) -> list[str]:
    return re.findall(
        r"```python\s*\n(.*?)```", response, flags=re.DOTALL | re.IGNORECASE
    )


def test_loaded_prompts_contain_sdk_separation_rules():
    """Both prompt sources must contain the SDK and search-language rules."""
    instructions = Path("instructions.md").read_text(encoding="utf-8")

    for prompt in (instructions, docs_agent_prompt):
        prompt_lower = prompt.lower()
        assert SDK_RULE.lower() in prompt_lower
        assert LANGUAGE_SEARCH_RULE in prompt_lower
        for method in JAVASCRIPT_METHODS:
            assert method.lower() in prompt_lower


@pytest.mark.langsmith
def test_python_context_response_has_no_javascript_methods_in_python_fences():
    """Python-context responses must not put JavaScript APIs in Python fences."""
    response = """Use the Python API for this graph:\n\n```python\nworkflow.add_node(\"agent\", agent)\nworkflow.add_edge(START, \"agent\")\n```\n"""
    python_blocks = _python_code_blocks(response)
    camel_case_methods = [
        method
        for block in python_blocks
        for method in JAVASCRIPT_METHODS
        if re.search(rf"\b{re.escape(method)}\b", block)
    ]

    t.log_inputs({"response": response})
    t.log_outputs(
        {"python_blocks": python_blocks, "camel_case_methods": camel_case_methods}
    )
    t.log_reference_outputs({"camel_case_methods": []})

    assert camel_case_methods == []
