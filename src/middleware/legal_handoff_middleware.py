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

_STRONG_LEGAL_PATTERNS = [
    # Data processing agreements
    r"\bdata processing (?:agreement|addendum|terms)s?\b",
    # Cross-border transfer terms
    r"\bstandard contractual clauses?\b",
    # Healthcare
    r"\bbusiness associate (?:agreement|addendum)s?\b",
    r"\bHIPAA\b",
    r"\bprotected health information\b",
    # Contracts
    r"\bmaster (?:services?|subscription) agreements?\b",
    r"\bcontract(?:ual)? (?:amendments?|terms|clauses?|redlines?|renewals?|negotiations?)\b",
    r"\b(?:amend(?:ment|ing)?|redline[sd]?|negotiat\w*|sign\w*) (?:the |our |my |your |a |an |this )?(?:contract|agreement|addendum)s?\b",
    r"\blegal (?:review|terms|team)\b",
    r"\bindemnif\w*\b",
    r"\blimitation of liability\b",
    r"\bliability caps?\b",
]

_AMBIGUOUS_LEGAL_PATTERNS = [
    r"\bDPAs?\b",
    r"\bSCCs?\b",
    r"\bBAAs?\b",
    r"\bMSAs?\b",
    r"\border forms?\b",
    r"\bredlines?\b",
    r"\bhealth (?:data|information|records)\b",
]

_STRONG_LEGAL_RE = re.compile("|".join(_STRONG_LEGAL_PATTERNS), re.IGNORECASE)
_AMBIGUOUS_LEGAL_RE = re.compile("|".join(_AMBIGUOUS_LEGAL_PATTERNS), re.IGNORECASE)
_LEGAL_CONTEXT_RE = re.compile(
    r"\b(?:contracts?|agreements?|addendums?|clauses?|legal|lawyers?|counsel|"
    r"sign(?:ed|ing)?|GDPR|data transfers?|cover(?:s|ed|ing|age)?|compliance|"
    r"customer terms)\b",
    re.IGNORECASE,
)

# Portal chats are not monitored; support only sees a conversation once the
# customer opens a ticket. Never promise a follow-up here, point to the ticket
# path instead: the Contact support button on the portal, or the KB article on
# surfaces without that button (e.g. docs).
SUPPORT_TICKET_ARTICLE_URL = (
    "https://support.langchain.com/articles/"
    "3788815465-how-to-open-support-tickets-via-portal#how-to-submit-a-support-ticket"
)

LEGAL_HANDOFF_MESSAGE = (
    "**I can't answer questions about legal or contract terms, so please open a "
    "support ticket and a person from our team will handle it.**\n\n"
    "That includes DPAs, SCCs, BAAs, HIPAA and health-data coverage, and contract "
    "amendments or redlines. I can't confirm what these documents cover or whether "
    "proposed changes are acceptable, and nothing I've said in this conversation "
    "should be read as legal confirmation.\n\n"
    "If you're on the support portal, click **Contact support** below this reply. "
    "Otherwise, follow the steps in "
    f"[How to Submit a Support Ticket]({SUPPORT_TICKET_ARTICLE_URL})."
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


def _legal_match(text: str) -> re.Match[str] | None:
    """Return a strong legal match or an ambiguous match with legal context."""
    match = _STRONG_LEGAL_RE.search(text)
    if match:
        return match
    if _LEGAL_CONTEXT_RE.search(text):
        return _AMBIGUOUS_LEGAL_RE.search(text)
    return None


def is_legal_question(text: str) -> bool:
    """Return whether the text asks about legal or contract terms."""
    return bool(_legal_match(text))


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

        match = _legal_match(text)
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


__all__ = [
    "LEGAL_HANDOFF_MESSAGE",
    "SUPPORT_TICKET_ARTICLE_URL",
    "LegalHandoffMiddleware",
    "is_legal_question",
]
