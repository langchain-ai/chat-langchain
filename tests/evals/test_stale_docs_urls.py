"""Tests for stale /docs/ URL prohibition in docs_agent_prompt.

Bug: The agent generates hallucinated URLs with /docs/ path prefix from training memory,
e.g. https://docs.langchain.com/docs/introduction/ which returns 404.
These differ from valid /oss/ and /langsmith/ paths that the Mintlify API returns.

Root cause: System prompt warns against python.langchain.com but not against
docs.langchain.com/docs/ prefix which is equally stale.

Affected traces (2026-03-15):
- 019cf020-41f8-7260-a64e-d42542717f7d  tell me about langchain
- 019cf020-91f6-76f2-ad3b-63ece98cc994  what is langchain
- 019cf023-8530-7fb1-b4a9-8455115d607f  events["event"] kinds
"""

import pytest
from langsmith import testing as t

from src.prompts.docs_agent_prompt import docs_agent_prompt


@pytest.mark.langsmith
def test_prompt_prohibits_docs_langchain_com_docs_prefix():
    """Prompt must explicitly warn against docs.langchain.com/docs/ path URLs.

    Valid LangChain documentation paths: /oss/..., /langsmith/..., /langgraph/...
    Stale 404 paths (from model training data): /docs/...

    The prompt already warns against python.langchain.com but not against
    the equally-broken docs.langchain.com/docs/ prefix.
    """
    t.log_inputs({"check": "docs_prefix_explicitly_prohibited"})

    # The prompt must contain an explicit mention of docs.langchain.com/docs/ as forbidden
    has_explicit_docs_warning = "docs.langchain.com/docs/" in docs_agent_prompt

    t.log_outputs({"has_explicit_docs_warning": has_explicit_docs_warning})
    t.log_reference_outputs({"has_explicit_docs_warning": True})

    assert has_explicit_docs_warning, (
        "System prompt must explicitly mention and prohibit docs.langchain.com/docs/ URLs. "
        "These are stale training-data paths that return 404. Three production traces on "
        "2026-03-15 show agent generating these broken links:\n"
        "- https://docs.langchain.com/docs/introduction/ (404)\n"
        "- https://docs.langchain.com/docs/components/ (404)\n"
        "- https://docs.langchain.com/docs/modules/agents/ (404)\n"
        "The prompt already prohibits python.langchain.com; the same fix is needed for /docs/ prefix."
    )


@pytest.mark.langsmith
def test_prompt_provides_correct_url_path_guidance():
    """Prompt must tell agent which URL path prefixes are valid for docs.langchain.com.

    Valid: /oss/python/..., /oss/javascript/..., /langsmith/..., /langgraph/...
    Invalid (stale): /docs/...
    """
    t.log_inputs({"check": "valid_path_prefixes_listed"})

    # The prompt already has /oss/python/ example — check it explicitly mentions the pattern
    has_oss_path_example = "docs.langchain.com/oss/" in docs_agent_prompt

    t.log_outputs({"has_oss_path_example": has_oss_path_example})
    t.log_reference_outputs({"has_oss_path_example": True})

    assert has_oss_path_example, (
        "System prompt must include an example of a valid docs.langchain.com/oss/ URL "
        "so the model knows to use /oss/ paths, not /docs/ paths."
    )


@pytest.mark.langsmith
def test_prompt_stale_url_section_covers_docs_prefix():
    """The stale URL warning section must cover docs.langchain.com/docs/ not just python.langchain.com.

    Currently the 'NEVER include links to python.langchain.com' section only covers
    that one domain. The /docs/ prefix on docs.langchain.com is a separate stale path.
    """
    t.log_inputs({"check": "stale_url_section_comprehensive"})

    # The stale URL warning section should mention both patterns
    has_python_langchain_warning = "python.langchain.com" in docs_agent_prompt
    has_docs_prefix_warning = "docs.langchain.com/docs/" in docs_agent_prompt

    t.log_outputs({
        "has_python_langchain_warning": has_python_langchain_warning,
        "has_docs_prefix_warning": has_docs_prefix_warning,
        "both_covered": has_python_langchain_warning and has_docs_prefix_warning,
    })
    t.log_reference_outputs({"both_covered": True})

    assert has_python_langchain_warning and has_docs_prefix_warning, (
        f"System prompt covers python.langchain.com={has_python_langchain_warning} "
        f"but docs.langchain.com/docs/ prefix={has_docs_prefix_warning}. "
        "Both stale URL patterns must be explicitly prohibited."
    )
