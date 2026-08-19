"""Pre-answer domain gate for trigger-sourced agent invocations.

Background
----------
A tenant deep-agent scoped to one domain (e.g. competitor intelligence) can be
wired to a passive trigger such as ``ACTION: gmail email received`` while its
instruction file only defines contracts for the surfaces it was authored for
(Slack, scheduled refreshes). With no declared contract for the trigger shape,
the tenant model falls back on the generic base identity and answers arbitrary
inbound email as a general-purpose assistant: summarizing design-tool and
issue-tracker notifications, offering to RSVP to invites, and restating
internal HR/payroll content — with zero domain tool calls and no scope notice.

This middleware adds the missing pre-answer check. Trigger-sourced payloads are
classified against the agent's declared domain *before* the agent is allowed to
produce a user-facing answer; a payload that does not match short-circuits to a
single-line, in-scope decline instead of a general-purpose answer. It mirrors
:mod:`src.middleware.guardrails_middleware` (classifier before the agent,
``jump_to="end"`` on BLOCK) but keys off the trigger payload rather than an
interactive user query, so ordinary chat turns are never gated by it.

``validate_trigger_wiring`` is the wiring-time half: it rejects (or warns about)
an email/gmail trigger attached to an agent whose instructions declare no
email-handling contract, which is the configuration that produced the incident.
"""
from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Iterable
from typing import Any, Literal, Protocol

from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired, TypedDict

logger = logging.getLogger(__name__)

CLASSIFIER_MAX_RETRIES = 2
CLASSIFIER_TIMEOUT_SECONDS = 10

# Trigger payloads are machine-generated envelopes, not user prose: Fleet-style
# triggers open with an ``ACTION: <trigger name>`` header line.
TRIGGER_HEADER_PATTERN = re.compile(r"^\s*ACTION:\s*(?P<name>.+)$", re.MULTILINE)

TRIGGER_METADATA_KEYS: tuple[str, ...] = ("source", "source_type", "invocation_source")
TRIGGER_METADATA_VALUES: frozenset[str] = frozenset({"trigger", "cron", "schedule"})

EMAIL_TRIGGER_MARKERS: tuple[str, ...] = (
    "gmail",
    "email",
    "outlook",
    "inbox",
    "mail received",
)

# An instruction file declares an email contract when it says what the agent
# does with an email-shaped invocation, not merely that it can send mail.
EMAIL_CONTRACT_MARKERS: tuple[str, ...] = (
    "email received",
    "email trigger",
    "gmail trigger",
    "gmail received",
    "inbound email",
    "email payload",
    "when an email arrives",
    "email-received",
)

TRIGGER_DOMAIN_CLASSIFIER_PROMPT = """You are a domain gate for a single-purpose agent. \
The agent was built for exactly one job:

<agent_domain>
{agent_domain}
</agent_domain>

The agent has been invoked by an automated trigger (an inbound email, a schedule, \
a channel event). You decide whether the trigger payload is work this agent exists \
to do. You are not answering the payload and you never summarize it.

## ALLOW - the payload is in domain:
- It concerns a subject the declared domain names (an entity, account, product, \
topic, or artifact the agent tracks).
- It is a request, question, or update that the agent's declared job covers, even \
if it is phrased loosely or arrives as a forwarded thread.
- It plausibly changes something the agent maintains for its domain.

## BLOCK - the payload is out of domain:
- Routine notifications from tools the agent does not own (design tools, issue \
trackers, CI, code review, document comments).
- Calendar invitations, meeting changes, cancellations, and RSVP requests.
- Social, scheduling, or logistics chatter.
- Internal company administration: HR, payroll, benefits, retirement accounts, \
expenses, IT announcements. BLOCK these even when they name the agent's own \
company — an agent scoped to public research must not ingest internal records.
- Newsletters, marketing blasts, and receipts with no bearing on the declared \
domain.
- Anything whose only connection to the domain is that it mentions a company name \
in a signature, footer, or recipient list.

## Critical rules:
1. The declared domain is the only thing that makes a payload in domain. General \
usefulness is irrelevant: "the user would probably like a summary of this" is a \
BLOCK, not an ALLOW.
2. When the payload is out of domain, BLOCK even if the agent could technically \
answer it.
3. When the payload is genuinely about the declared domain but thin on detail, \
ALLOW and let the agent do its job.

Return the decision and one concise sentence naming the policy reason."""

DEFAULT_DECLINE_TEMPLATE = (
    "This trigger is outside my scope ({agent_domain}), so I'm not acting on it."
)


class TriggerDomainDecision(TypedDict):
    """Structured output for the trigger domain classifier."""

    decision: Literal["ALLOW", "BLOCK"]
    explanation: str


class TriggerContractError(Exception):
    """Raised when a trigger is wired to an agent that declares no contract for it."""

    pass


class TriggerDomainState(AgentState):
    """Extended state schema with the off-domain trigger flag."""

    off_domain_trigger: NotRequired[bool]


class TriggerDomainClassifier(Protocol):
    """Callable that classifies a trigger payload against the agent's domain."""

    async def __call__(self, payload: str) -> TriggerDomainDecision: ...


def parse_trigger_name(text: str) -> str | None:
    """Return the trigger name from an ``ACTION:`` payload header, if present."""
    match = TRIGGER_HEADER_PATTERN.search(text or "")
    return match.group("name").strip() if match else None


def is_trigger_payload(text: str, metadata: dict[str, Any] | None = None) -> bool:
    """Return whether this invocation came from an automated trigger."""
    for key in TRIGGER_METADATA_KEYS:
        value = (metadata or {}).get(key)
        if isinstance(value, str) and value.strip().lower() in TRIGGER_METADATA_VALUES:
            return True
    return parse_trigger_name(text) is not None


def is_email_trigger(trigger_name: str) -> bool:
    """Return whether a trigger name identifies an email/gmail invocation path."""
    lowered = (trigger_name or "").lower()
    return any(marker in lowered for marker in EMAIL_TRIGGER_MARKERS)


def declares_email_contract(instructions: str) -> bool:
    """Return whether instructions define a contract for email-shaped invocations."""
    lowered = (instructions or "").lower()
    return any(marker in lowered for marker in EMAIL_CONTRACT_MARKERS)


def validate_trigger_wiring(
    triggers: Iterable[str],
    instructions: str,
    *,
    strict: bool = False,
) -> list[str]:
    """Report email triggers wired to instructions that declare no email contract."""
    problems: list[str] = []
    if declares_email_contract(instructions):
        return problems

    for trigger in triggers:
        if not is_email_trigger(trigger):
            continue
        problems.append(
            f"Trigger {trigger!r} delivers email payloads, but the agent's "
            "instructions declare no email-handling contract. Add a trigger "
            "contract section covering the email path (what counts as in scope, "
            "and that an out-of-scope email is a no-op with a single scope note "
            "and no summary) before wiring this trigger."
        )

    if problems and strict:
        raise TriggerContractError(" ".join(problems))
    for problem in problems:
        logger.warning(problem)
    return problems


class TriggerDomainGateMiddleware(AgentMiddleware[TriggerDomainState]):
    """Block trigger payloads that fall outside the agent's declared domain."""

    state_schema = TriggerDomainState

    def __init__(
        self,
        agent_domain: str,
        model: str | None = None,
        fallback_model: str | None = None,
        block_off_domain: bool = True,
        classifier: TriggerDomainClassifier | None = None,
        decline_message: str | None = None,
    ):
        """Initialize the gate with the agent's declared domain and a classifier."""
        super().__init__()
        if not agent_domain or not agent_domain.strip():
            raise TriggerContractError(
                "TriggerDomainGateMiddleware requires a declared agent_domain; a "
                "gate with no domain cannot classify trigger payloads."
            )
        self.agent_domain = agent_domain.strip()
        self.block_off_domain = block_off_domain
        self.decline_message = decline_message or DEFAULT_DECLINE_TEMPLATE.format(
            agent_domain=self.agent_domain
        )
        self._classifier = classifier
        self.classifier_llms: list[tuple[str, Any]] = []
        if classifier is None:
            if model is None:
                raise TriggerContractError(
                    "TriggerDomainGateMiddleware requires either a model or a classifier."
                )
            self.classifier_llms.append((model, init_chat_model(model=model, temperature=0)))
            if fallback_model and fallback_model != model:
                self.classifier_llms.append(
                    (fallback_model, init_chat_model(model=fallback_model, temperature=0))
                )

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(
        self, state: TriggerDomainState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Gate trigger payloads on the agent's declared domain before it answers."""
        payload = self._latest_human_text(state.get("messages", []))
        if payload is None:
            return None

        if not is_trigger_payload(payload, self._runtime_metadata(runtime)):
            return None

        try:
            decision = await self._classify(payload)
        except Exception as e:
            # Trigger runs are non-interactive: an unclassified payload must not
            # reach the tenant model as a general-purpose request, so fail closed.
            logger.error("Trigger domain classification failed; blocking payload: %s", e)
            decision = {
                "decision": "BLOCK",
                "explanation": "Domain classification failed; failing closed.",
            }

        if decision["decision"] == "ALLOW":
            logger.info("Trigger payload in domain: %s", decision["explanation"])
            return None

        logger.warning(
            "Off-domain trigger payload for %r: %s",
            self.agent_domain,
            decision["explanation"],
        )
        if not self.block_off_domain:
            return None

        return {
            "messages": [AIMessage(content=self.decline_message)],
            "off_domain_trigger": True,
            "jump_to": "end",
        }

    async def _classify(self, payload: str) -> TriggerDomainDecision:
        """Classify a trigger payload as ALLOW or BLOCK for the declared domain."""
        if self._classifier is not None:
            return await self._classifier(payload)

        prompt = [
            SystemMessage(
                content=TRIGGER_DOMAIN_CLASSIFIER_PROMPT.format(
                    agent_domain=self.agent_domain
                )
            ),
            HumanMessage(
                content=(
                    "Classify this trigger payload for the agent described above. "
                    "Do not answer or summarize it.\n\nTrigger payload:\n"
                    f"{payload}"
                )
            ),
        ]

        last_exception: Exception | None = None
        for model_name, llm in self.classifier_llms:
            structured_llm = llm.with_structured_output(TriggerDomainDecision)
            for attempt in range(CLASSIFIER_MAX_RETRIES + 1):
                try:
                    return await asyncio.wait_for(
                        structured_llm.ainvoke(
                            prompt, config={"tags": ["trigger-domain-gate"]}
                        ),
                        timeout=CLASSIFIER_TIMEOUT_SECONDS,
                    )
                except Exception as e:
                    last_exception = e
                    logger.warning(
                        "Trigger domain classification failed with %s attempt %s/%s: %s",
                        model_name,
                        attempt + 1,
                        CLASSIFIER_MAX_RETRIES + 1,
                        e,
                    )

        raise RuntimeError(
            f"Trigger domain classification failed after retries: {last_exception}"
        )

    def _runtime_metadata(self, runtime: Runtime) -> dict[str, Any]:
        """Return run metadata from the runtime, tolerating missing context."""
        for attribute in ("context", "config"):
            container = getattr(runtime, attribute, None)
            if isinstance(container, dict):
                metadata = container.get("metadata")
                if isinstance(metadata, dict):
                    return metadata
                return container
        return {}

    def _latest_human_text(self, messages: list) -> str | None:
        """Return the text of the most recent human message, if any."""
        for message in reversed(messages):
            if not isinstance(message, HumanMessage):
                continue
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = [
                    block if isinstance(block, str) else block.get("text", "")
                    for block in content
                    if isinstance(block, str)
                    or (isinstance(block, dict) and block.get("type") == "text")
                ]
                return " ".join(part for part in parts if part).strip()
            return str(content)
        return None


__all__ = [
    "EMAIL_CONTRACT_MARKERS",
    "TRIGGER_DOMAIN_CLASSIFIER_PROMPT",
    "TriggerContractError",
    "TriggerDomainDecision",
    "TriggerDomainGateMiddleware",
    "declares_email_contract",
    "is_email_trigger",
    "is_trigger_payload",
    "parse_trigger_name",
    "validate_trigger_wiring",
]
