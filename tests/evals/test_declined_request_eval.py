"""Tests verifying that the 'declined-request' LangSmith auto-evaluator never
crashes on errored agent runs.

Root cause
----------
The ``declined-request`` online auto-evaluator rule fires on **every** run in
the Chat-LangChain production project, including runs that raised an exception
and therefore have ``outputs=null``.  When the LangSmith rule tries to evaluate
such a run it passes ``{input, examples_few_shot}`` to the ``StructuredPrompt``
but omits ``{output}`` (because ``outputs`` is ``null``).  The prompt template
requires all three variables, so the evaluator crashes with::

    KeyError: "Input to StructuredPrompt is missing variables {'output'}.
               Expected: ('examples_few_shot', 'input', 'output')
               Received: ('examples_few_shot', 'input')"

Fix
---
``GuardrailsMiddleware`` now eagerly emits a ``declined-request`` LangSmith
feedback score from ``abefore_agent`` (immediately after the guardrails
classification decision).  Because this score is written *before* the agent
loop begins, it is present on the run record even when the agent later errors.
The LangSmith evaluator rule sees the pre-existing score and skips
re-evaluation, preventing the ``KeyError``.

As a belt-and-suspenders measure ``aafter_agent`` also posts a sentinel score
of ``None`` when the agent produced no AI output, covering the case where
``abefore_agent`` could not emit a score (e.g. due to a transient network error).

Trace
-----
Failing evaluator run: ``019af2b3-cbbb-7b1f-8065-4e5cf57ba29d``
Production agent run:  ``019ae939-0c82-7554-9597-855c7170ea80`` (errored)
Project:               Chat-LangChain (``dcffe24f-52f0-434f-aa22-932d27cb23ef``)
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import testing as t

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MW_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "src", "middleware", "guardrails_middleware.py"
)


def _read_middleware_source() -> str:
    with open(_MW_PATH) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Static source-level checks
# ---------------------------------------------------------------------------


@pytest.mark.langsmith
def test_aafter_agent_hook_exists():
    """GuardrailsMiddleware must define aafter_agent so errored runs get a sentinel score."""
    source = _read_middleware_source()

    t.log_inputs({"module_path": _MW_PATH})
    t.log_outputs({"has_aafter_agent": "aafter_agent" in source})

    assert "aafter_agent" in source, (
        "GuardrailsMiddleware must implement aafter_agent to guard against "
        "the 'declined-request' evaluator crashing on errored runs "
        "(trace 019af2b3-cbbb-7b1f-8065-4e5cf57ba29d)"
    )


@pytest.mark.langsmith
def test_declined_request_feedback_key_defined():
    """The _DECLINED_REQUEST_FEEDBACK_KEY constant must match the LangSmith rule key."""
    source = _read_middleware_source()

    t.log_inputs({"module_path": _MW_PATH})
    t.log_outputs({"has_key_constant": "_DECLINED_REQUEST_FEEDBACK_KEY" in source})

    assert "_DECLINED_REQUEST_FEEDBACK_KEY" in source, (
        "guardrails_middleware.py must define _DECLINED_REQUEST_FEEDBACK_KEY "
        "so the feedback key stays in sync with the LangSmith evaluator rule"
    )
    assert '"declined-request"' in source or "'declined-request'" in source, (
        "_DECLINED_REQUEST_FEEDBACK_KEY must be set to 'declined-request' "
        "to match the LangSmith online auto-evaluator rule key"
    )


@pytest.mark.langsmith
def test_emit_declined_request_feedback_method_exists():
    """_emit_declined_request_feedback must exist and call create_feedback."""
    source = _read_middleware_source()

    t.log_inputs({"module_path": _MW_PATH})
    t.log_outputs({"has_method": "_emit_declined_request_feedback" in source})

    assert "_emit_declined_request_feedback" in source, (
        "GuardrailsMiddleware must have _emit_declined_request_feedback() "
        "to write the 'declined-request' score to LangSmith before the agent runs"
    )
    assert "create_feedback" in source, (
        "guardrails_middleware.py must call ls.Client().create_feedback() "
        "to persist the 'declined-request' score"
    )


@pytest.mark.langsmith
def test_abefore_agent_emits_score_for_allowed_queries():
    """abefore_agent must call _emit_declined_request_feedback for ALLOWED decisions."""
    source = _read_middleware_source()

    t.log_inputs({"module_path": _MW_PATH})
    # Check that _emit_declined_request_feedback appears after ALLOWED handling logic
    allowed_idx = source.find('decision == "ALLOWED"')
    emit_after_allowed = source.find("_emit_declined_request_feedback", allowed_idx)

    t.log_outputs({
        "allowed_idx": allowed_idx,
        "emit_after_allowed": emit_after_allowed,
        "found": allowed_idx != -1 and emit_after_allowed != -1,
    })

    assert allowed_idx != -1, "abefore_agent must handle the ALLOWED decision"
    assert emit_after_allowed != -1, (
        "abefore_agent must call _emit_declined_request_feedback() after the "
        "ALLOWED decision so errored runs already have a score before the agent "
        "processes the request"
    )


@pytest.mark.langsmith
def test_abefore_agent_emits_score_for_blocked_queries():
    """abefore_agent must call _emit_declined_request_feedback for BLOCKED decisions."""
    source = _read_middleware_source()

    t.log_inputs({"module_path": _MW_PATH})
    # Check that _emit_declined_request_feedback appears before jump_to end
    emit_before_jump = (
        source.find("_emit_declined_request_feedback")
        < source.find('"jump_to": "end"')
    )

    t.log_outputs({"emit_before_jump": emit_before_jump})

    assert emit_before_jump, (
        "abefore_agent must call _emit_declined_request_feedback() *before* "
        "returning jump_to=end so the BLOCKED decision is persisted even if "
        "the rejection message generation raises an exception"
    )


# ---------------------------------------------------------------------------
# Unit tests (mock LangSmith client)
# ---------------------------------------------------------------------------


def _make_middleware():
    """Return a GuardrailsMiddleware with a mocked LLM."""
    from src.middleware.guardrails_middleware import GuardrailsMiddleware

    mw = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    mw.llm = MagicMock()
    mw.block_off_topic = True
    return mw


@pytest.mark.langsmith
def test_emit_declined_request_feedback_calls_create_feedback():
    """_emit_declined_request_feedback should call ls.Client().create_feedback."""
    from src.middleware.guardrails_middleware import (
        GuardrailsMiddleware,
        _DECLINED_REQUEST_FEEDBACK_KEY,
    )

    run_tree_mock = MagicMock()
    run_tree_mock.id = "test-run-id-1234"
    client_mock = MagicMock()

    mw = _make_middleware()

    t.log_inputs({"declined": True, "comment": "test comment"})

    with (
        patch("langsmith.get_current_run_tree", return_value=run_tree_mock),
        patch("langsmith.Client", return_value=client_mock),
    ):
        mw._emit_declined_request_feedback(declined=True, comment="test comment")

    client_mock.create_feedback.assert_called_once()
    call_kwargs = client_mock.create_feedback.call_args[1]

    t.log_outputs({
        "run_id": call_kwargs.get("run_id"),
        "key": call_kwargs.get("key"),
        "score": call_kwargs.get("score"),
    })

    assert call_kwargs["run_id"] == "test-run-id-1234"
    assert call_kwargs["key"] == _DECLINED_REQUEST_FEEDBACK_KEY
    assert call_kwargs["score"] == 1  # declined=True -> score=1


@pytest.mark.langsmith
def test_emit_declined_request_feedback_score_0_for_allowed():
    """_emit_declined_request_feedback with declined=False should write score=0."""
    from src.middleware.guardrails_middleware import (
        GuardrailsMiddleware,
        _DECLINED_REQUEST_FEEDBACK_KEY,
    )

    run_tree_mock = MagicMock()
    run_tree_mock.id = "test-run-allowed"
    client_mock = MagicMock()

    mw = _make_middleware()

    t.log_inputs({"declined": False})

    with (
        patch("langsmith.get_current_run_tree", return_value=run_tree_mock),
        patch("langsmith.Client", return_value=client_mock),
    ):
        mw._emit_declined_request_feedback(declined=False)

    call_kwargs = client_mock.create_feedback.call_args[1]

    t.log_outputs({"score": call_kwargs.get("score")})

    assert call_kwargs["score"] == 0, (
        "An allowed query should emit score=0, not 1"
    )


@pytest.mark.langsmith
def test_emit_declined_request_feedback_noop_without_run_tree():
    """_emit_declined_request_feedback must not raise when there is no active run tree."""
    from src.middleware.guardrails_middleware import GuardrailsMiddleware

    mw = _make_middleware()

    t.log_inputs({"run_tree": None})

    with patch("langsmith.get_current_run_tree", return_value=None):
        # Should not raise
        mw._emit_declined_request_feedback(declined=False)

    t.log_outputs({"raised": False})


@pytest.mark.langsmith
def test_aafter_agent_posts_sentinel_when_no_ai_output():
    """aafter_agent must post score=None when messages contain no AIMessage.

    This is the key guard for errored runs: if the agent errors before
    producing any output, the 'declined-request' LangSmith auto-evaluator
    would crash with KeyError on the missing {output} template variable.
    The aafter_agent hook posts a sentinel score so the evaluator sees an
    existing score and skips re-evaluation.
    """
    from src.middleware.guardrails_middleware import (
        _DECLINED_REQUEST_FEEDBACK_KEY,
    )

    run_tree_mock = MagicMock()
    run_tree_mock.id = "test-errored-run"
    client_mock = MagicMock()

    mw = _make_middleware()
    state = {"messages": [HumanMessage(content="How do I use LangChain?")]}
    runtime = MagicMock()

    t.log_inputs({"messages_count": len(state["messages"]), "has_ai_message": False})

    with (
        patch("langsmith.get_current_run_tree", return_value=run_tree_mock),
        patch("langsmith.Client", return_value=client_mock),
    ):
        asyncio.run(mw.aafter_agent(state, runtime))

    client_mock.create_feedback.assert_called_once()
    call_kwargs = client_mock.create_feedback.call_args[1]

    t.log_outputs({
        "key": call_kwargs.get("key"),
        "score": call_kwargs.get("score"),
        "comment": call_kwargs.get("comment"),
    })

    assert call_kwargs["key"] == _DECLINED_REQUEST_FEEDBACK_KEY
    assert call_kwargs["score"] is None, (
        "Errored run (no AI output) must emit score=None so the auto-evaluator "
        "knows to skip this run rather than crash on the missing {output} variable"
    )
    assert "errored" in (call_kwargs.get("comment") or "").lower() or \
           "no output" in (call_kwargs.get("comment") or "").lower(), (
        "The sentinel feedback comment should explain why evaluation was skipped"
    )


@pytest.mark.langsmith
def test_aafter_agent_noop_when_ai_output_present():
    """aafter_agent must NOT post sentinel feedback when AI output is present.

    When the agent completed normally (produced at least one AIMessage), the
    abefore_agent hook has already posted a definitive score.  aafter_agent
    should be a no-op in this case.
    """
    run_tree_mock = MagicMock()
    run_tree_mock.id = "test-success-run"
    client_mock = MagicMock()

    mw = _make_middleware()
    state = {
        "messages": [
            HumanMessage(content="How do I use LangChain?"),
            AIMessage(content="Here is how you use LangChain..."),
        ]
    }
    runtime = MagicMock()

    t.log_inputs({"messages_count": len(state["messages"]), "has_ai_message": True})

    with (
        patch("langsmith.get_current_run_tree", return_value=run_tree_mock),
        patch("langsmith.Client", return_value=client_mock),
    ):
        result = asyncio.run(mw.aafter_agent(state, runtime))

    t.log_outputs({
        "create_feedback_called": client_mock.create_feedback.called,
        "return_value": result,
    })

    client_mock.create_feedback.assert_not_called(), (
        "aafter_agent must not post feedback when AI output is present "
        "(abefore_agent already posted a definitive score)"
    )
    assert result is None


@pytest.mark.langsmith
def test_aafter_agent_noop_without_run_tree():
    """aafter_agent must not raise when there is no active LangSmith run tree."""
    mw = _make_middleware()
    # State with no AI output to trigger the feedback path
    state = {"messages": [HumanMessage(content="test")]}
    runtime = MagicMock()

    t.log_inputs({"run_tree": None})

    with patch("langsmith.get_current_run_tree", return_value=None):
        # Should not raise
        result = asyncio.run(mw.aafter_agent(state, runtime))

    t.log_outputs({"raised": False, "result": result})
    assert result is None


@pytest.mark.langsmith
def test_declined_request_key_value():
    """The _DECLINED_REQUEST_FEEDBACK_KEY must equal 'declined-request' (exact match).

    This value must match the 'key' field configured in the LangSmith
    automation rule (rule ID ab4a7e90-1c82-4b09-b659-e1d7308c0abe).
    """
    from src.middleware.guardrails_middleware import _DECLINED_REQUEST_FEEDBACK_KEY

    t.log_inputs({})
    t.log_outputs({"key": _DECLINED_REQUEST_FEEDBACK_KEY})

    assert _DECLINED_REQUEST_FEEDBACK_KEY == "declined-request", (
        f"Expected 'declined-request', got {_DECLINED_REQUEST_FEEDBACK_KEY!r}. "
        "This key must match the LangSmith automation rule configuration exactly."
    )
