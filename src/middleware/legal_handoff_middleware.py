"""Hand legal and contract questions to a person instead of answering them.

The docs agent has no authority to confirm what a DPA, SCC, BAA, or contract
covers, or whether a customer's amendment is acceptable. Answers it produced
from support articles read as legal confirmation (e.g. "the DPA covers health
data"). This middleware runs before the guardrails classifier and the model,
matches those topics deterministically, and returns a fixed handoff message so
no model output can confirm or interpret legal terms.

Detection is code rather than a classifier prompt because the production
guardrails prompt is pulled from the LangSmith hub.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import langsmith as ls
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

logger = logging.getLogger(__name__)

_LEGAL_PATTERNS = [
    # Data processing agreements
    r"\bDPAs?\b",
    r"\bdata processing (?:agreement|addendum|terms)s?\b",
    # Cross-border transfer terms
    r"\bSCCs?\b",
    r"\bstandard contractual clauses?\b",
    # Healthcare
    r"\bBAAs?\b",
    r"\bbusiness associate (?:agreement|addendum)s?\b",
    r"\bHIPAA\b",
    r"\bprotected health information\b",
    r"\bhealth (?:data|information|records)\b",
    # Contracts
    r"\bMSAs?\b",
    r"\bmaster (?:services?|subscription) agreements?\b",
    r"\border forms?\b",
    r"\bcontract(?:ual)? (?:amendments?|terms|clauses?|redlines?|renewals?|negotiations?)\b",
    r"\b(?:amend(?:ment|ing)?|redline[sd]?|negotiat\w*|sign\w*) (?:the |our |my |your |a |an |this )?(?:contract|agreement|addendum)s?\b",
    r"\bredlines?\b",
    r"\blegal (?:review|terms|team)\b",
    r"\bindemnif\w*\b",
    r"\blimitation of liability\b",
    r"\bliability caps?\b",
]

_LEGAL_RE = re.compile("|".join(_LEGAL_PATTERNS), re.IGNORECASE)

LEGAL_HANDOFF_MESSAGE = (
    "**I can't answer questions about legal or contract terms, so a person from "
    "our team needs to handle this.**\n\n"
    "That includes DPAs, SCCs, BAAs, HIPAA and health-data coverage, and contract "
    "amendments or redlines. I can't confirm what these documents cover or whether "
    "proposed changes are acceptable, and nothing I've said in this conversation "
    "should be read as legal confirmation. A member of the LangChain team will "
    "follow up with you on this."
)


class LegalHandoffState(AgentState):
    """Extended state schema with the legal handoff flag."""

    legal_handoff: NotRequired[bool]


def _message_text(content: Any) -> str:
    """Return only the text blocks of a message's content."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = [
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str)
        or (isinstance(block, dict) and block.get("type") == "text")
    ]
    return " ".join(part for part in parts if isinstance(part, str))


def is_legal_question(text: str) -> bool:
    """Return whether the text asks about legal or contract terms."""
    return bool(_LEGAL_RE.search(text))


class LegalHandoffMiddleware(AgentMiddleware[LegalHandoffState]):
    """Return a fixed handoff for legal and contract questions."""

    state_schema = LegalHandoffState

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: LegalHandoffState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """End the run with a handoff when the latest user turn is a legal question."""
        messages = state.get("messages", [])
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                text = _message_text(message.content)
                break
        else:
            return None

        match = _LEGAL_RE.search(text)
        if not match:
            return None

        logger.info("Legal handoff triggered by term: %s", match.group(0))
        self._track_handoff_metadata(match.group(0))
        return {
            "messages": [AIMessage(content=LEGAL_HANDOFF_MESSAGE)],
            "legal_handoff": True,
            "jump_to": "end",
        }

    def _track_handoff_metadata(self, term: str) -> None:
        """Mark the run so support can find handed-off conversations."""
        try:
            run_tree = ls.get_current_run_tree()
            if run_tree:
                run_tree.metadata["legal_handoff"] = True
                run_tree.metadata["legal_handoff_term"] = term
        except Exception:
            pass


__all__ = ["LEGAL_HANDOFF_MESSAGE", "LegalHandoffMiddleware", "is_legal_question"]
