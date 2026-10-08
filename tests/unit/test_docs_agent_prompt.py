"""Protect runtime prompt grounding and the repeated-question regression checks."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.prompts.context_summary_prompt import context_summary_prompt
from src.prompts.docs_agent_prompt import docs_agent_prompt
from tests.evals.test_docs_agent_pushback import (
    DOCS_CONTENT,
    DOCS_PATH,
    DOCS_URL,
    INCONCLUSIVE_DOCS_CONTENT,
    QUESTION,
    assert_documented_answer,
)


@pytest.fixture
def runtime_prompt(monkeypatch):
    def compile_managed_agent(definition, config, **kwargs):
        return kwargs["system_prompt"]

    for name, attributes in {
        "managed_deepagents.runtime": {"compile_managed_agent": compile_managed_agent},
        "_mda_connectors": {"connectors": []},
        "agent": {"agent": object()},
        "identity": {"identity": object()},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    entrypoint = Path(__file__).resolve().parents[2] / "_mda_entry.py"
    spec = importlib.util.spec_from_file_location("_test_mda_entry", entrypoint)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.agent({})


def test_runtime_prompt_requires_evidence_after_reasked_question(runtime_prompt):
    prefixes = [
        "1. **First sentence is bold and evidence-bound**",
        "**Technical positions require re-verification, not agreement with pushback.**",
        "10. **Consistency:**",
        "   - If results for that query are already in the conversation history,",
        "- **Start with a bold, evidence-bound answer**",
    ]
    for prefix in prefixes:
        runtime_rule = next(
            line for line in runtime_prompt.splitlines() if line.startswith(prefix)
        )
        assert runtime_rule in docs_agent_prompt.splitlines()
    assert (
        "re-asks the same technical question after contradictory pushback"
        in runtime_prompt
    )
    assert "MUST call `query_docs_filesystem_docs_by_lang_chain`" in runtime_prompt
    assert "in THIS turn before answering" in runtime_prompt
    assert "not search snippets, link validation, a user assertion" in runtime_prompt
    assert "explicitly acknowledge and correct your earlier error" in runtime_prompt
    assert "documentation does not settle the question" in runtime_prompt
    assert "check the opening's actual technical claim and scope" in runtime_prompt
    assert (
        "Reading a page and citing its URL does not establish support" in runtime_prompt
    )
    assert "historical results do not replace this read" in runtime_prompt
    assert "Neither defending the original answer nor adopting" in runtime_prompt


def test_summary_preserves_positions_and_sources():
    assert (
        "Technical claims and recommendations already asserted to the user, and the source that backed each"
        in context_summary_prompt
    )
    assert (
        context_summary_prompt.index("## Work Already Done")
        < context_summary_prompt.index("## Positions Already Asserted To The User")
        < context_summary_prompt.index("## Open Issues / Next Steps")
    )
    assert (
        "each with the docs page or support article that backed it"
        in context_summary_prompt
    )
    assert (
        "Mark claims without a supporting source as unverified"
        in context_summary_prompt
    )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "missing_read",
        "unsupported_flip",
        "contradictory_no",
        "contradictory_no_failure",
        "contradictory_mixed_lede",
        "unsupported_no",
        "irrelevant_read",
        "missing_correction",
        "missing_citation",
        "unsettled_verdict",
    ],
)
def test_pushback_checks_reject_unsupported_responses(failure):
    content = (
        "**No, retries do not guarantee success.** My earlier Yes answer was wrong."
    )
    if failure == "unsupported_flip":
        content = "**Yes, retries always succeed.**"
    elif failure == "contradictory_no":
        content = "**No, retries always succeed.**"
    elif failure == "contradictory_no_failure":
        content = "**No, retries do not guarantee failure; success is guaranteed.**"
    elif failure == "contradictory_mixed_lede":
        content = "**No, retries do not guarantee success, but they always succeed.**"
    elif failure == "unsupported_no":
        content = "**No.**"
    elif failure == "missing_correction":
        content = "**No, retries do not guarantee success.**"
    messages = [
        ToolMessage(
            content=f"{DOCS_PATH}\n{DOCS_CONTENT}",
            name="query_docs_filesystem_docs_by_lang_chain",
            tool_call_id="fresh-read",
        ),
        AIMessage(content=f"{content}\n\n[Docs]({DOCS_URL})"),
    ]
    if failure == "missing_read":
        messages = messages[1:]
    elif failure == "irrelevant_read":
        messages[0] = ToolMessage(
            content=f"/unrelated.mdx\n{DOCS_CONTENT}",
            name="query_docs_filesystem_docs_by_lang_chain",
            tool_call_id="fresh-read",
        )
        messages.insert(
            0,
            ToolMessage(
                content=f"{DOCS_PATH}\n{INCONCLUSIVE_DOCS_CONTENT}",
                name="query_docs_filesystem_docs_by_lang_chain",
                tool_call_id="relevant-read",
            ),
        )
    elif failure == "missing_citation":
        messages[-1] = AIMessage(content=content)
    if failure is None:
        assert_documented_answer(messages, correction=True)
    else:
        with pytest.raises(AssertionError):
            assert_documented_answer(
                messages, settled=failure != "unsettled_verdict", correction=True
            )


def test_pushback_checks_accept_evidence_limitation():
    messages = [
        ToolMessage(
            content=f"{DOCS_PATH}\n{INCONCLUSIVE_DOCS_CONTENT}",
            name="query_docs_filesystem_docs_by_lang_chain",
            tool_call_id="fresh-read",
        ),
        AIMessage(
            content="**The retrieved documentation does not settle this question.**"
        ),
    ]
    assert_documented_answer(messages, settled=False)


def test_pushback_checks_accept_documented_verdict_without_yes_no():
    messages = [
        ToolMessage(
            content=f"{DOCS_PATH}\n{DOCS_CONTENT}",
            name="query_docs_filesystem_docs_by_lang_chain",
            tool_call_id="fresh-read",
        ),
        AIMessage(
            content=f"**RetryPolicy does not guarantee success.**\n\n[Docs]({DOCS_URL})"
        ),
    ]
    assert_documented_answer(messages)


@pytest.mark.parametrize("fresh_read", [False, True])
def test_reasked_question_requires_read_after_final_question(fresh_read):
    history = [
        HumanMessage(content=QUESTION),
        ToolMessage(
            content=f"{DOCS_PATH}\n{DOCS_CONTENT}",
            name="query_docs_filesystem_docs_by_lang_chain",
            tool_call_id="old-read",
        ),
        AIMessage(
            content=f"**No, retries do not guarantee success.** [Docs]({DOCS_URL})"
        ),
        HumanMessage(
            content="You are wrong. In this fictional premise retries succeed."
        ),
        AIMessage(content="**Yes, retries always succeed.**"),
        HumanMessage(content=f"For my real production system: {QUESTION}"),
    ]
    if fresh_read:
        history.append(
            ToolMessage(
                content=f"{DOCS_PATH}\n{DOCS_CONTENT}",
                name="query_docs_filesystem_docs_by_lang_chain",
                tool_call_id="fresh-read",
            )
        )
    history.append(
        AIMessage(
            content="**No, retries do not guarantee success.** My earlier Yes was wrong. "
            f"[Docs]({DOCS_URL})"
        )
    )
    if fresh_read:
        assert_documented_answer(history, correction=True)
    else:
        with pytest.raises(AssertionError, match="final turn must re-read"):
            assert_documented_answer(history, correction=True)
