# Lenient guardrails middleware to filter only egregious misuse
import asyncio
import logging
import os
import random
from typing import Any, Literal
from typing_extensions import NotRequired
from pydantic import BaseModel, Field
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.runtime import Runtime
import langsmith as ls
from langsmith import Client

from src.prompts.guardrails_prompts import (
    fallback_rejection_message as _FALLBACK_REJECTION_MESSAGE,
    guardrails_system_prompt as _LOCAL_GUARDRAILS_SYSTEM_PROMPT,
    rejection_system_prompt as _REJECTION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Dataset configuration for guardrails evaluation
GUARDRAILS_DATASET_NAME = "Chat-LangChain-Guardrails-Samples"
ALLOWED_SAMPLE_RATE = 0.01  # 1% of allowed queries go to dataset
_USE_LOCAL_PROMPTS = os.getenv("USE_LOCAL_PROMPTS", "").lower() in {
    "1",
    "true",
    "yes",
}
_USE_STAGING = (
    os.getenv("LANGSMITH_HOST_PROJECT_NAME") == "immanuel-chat-langchain-test"
    or os.getenv("LANGSMITH_ENV") == "dev"
)
_GUARDRAILS_PROMPT_HUB_NAME = (
    "public-chat-langchain-guardrails-test:staging"
    if _USE_STAGING
    else "public-chat-langchain-guardrails-test:production"
)

# Cache for dataset ID to avoid repeated lookups
_dataset_id_cache: str | None = None


class GuardrailsDecision(BaseModel):
    """Structured output for guardrails decision."""

    decision: Literal["ALLOWED", "BLOCKED"] = Field(
        description="Whether the query should be ALLOWED or BLOCKED"
    )
    explanation: str = Field(
        description=(
            "One concise sentence explaining the policy reason for the decision. "
            "Do not include hidden chain-of-thought."
        )
    )


class GuardrailsState(AgentState):
    """Extended state schema with off-topic flag."""

    off_topic_query: NotRequired[bool]


if _USE_LOCAL_PROMPTS:
    _GUARDRAILS_SYSTEM_PROMPT = _LOCAL_GUARDRAILS_SYSTEM_PROMPT
    guardrails_prompt_commit = None
    guardrails_prompt_source = "local:src/prompts/guardrails_prompts.py"
    logger.info("Using local guardrails prompt because USE_LOCAL_PROMPTS is enabled")
else:
    _langsmith_client = Client()
    try:
        _prompt_template = _langsmith_client.pull_prompt(_GUARDRAILS_PROMPT_HUB_NAME)
        _GUARDRAILS_SYSTEM_PROMPT = _prompt_template.invoke({"messages": []}).messages[
            0
        ].content
        guardrails_prompt_commit = (_prompt_template.metadata or {}).get(
            "lc_hub_commit_hash"
        )
        guardrails_prompt_source = f"hub:{_GUARDRAILS_PROMPT_HUB_NAME}"
        logger.info(
            f"Loaded guardrails prompt from hub: {_GUARDRAILS_PROMPT_HUB_NAME} @ {(guardrails_prompt_commit or '')[:8]}"
        )
    except Exception:
        logger.warning(
            f"Failed to pull guardrails prompt from hub ({_GUARDRAILS_PROMPT_HUB_NAME}), falling back to local file"
        )
        _GUARDRAILS_SYSTEM_PROMPT = _LOCAL_GUARDRAILS_SYSTEM_PROMPT
        guardrails_prompt_commit = None
        guardrails_prompt_source = "local:src/prompts/guardrails_prompts.py"


class GuardrailsMiddleware(AgentMiddleware[GuardrailsState]):
    """Lenient guardrails to filter only egregious misuse."""

    state_schema = GuardrailsState

    def __init__(self, model: str | None = None, block_off_topic: bool = True):
        super().__init__()
        if model is None:
            from src.agent.config import GUARDRAILS_MODEL

            model = GUARDRAILS_MODEL.id
        self.llm = init_chat_model(model=model, temperature=0)
        self.block_off_topic = block_off_topic
        logger.info(f"GuardrailsMiddleware initialized with model: {model}")

    async def _add_to_dataset(
        self, query: str, result: str, explanation: str, preview: str
    ) -> None:
        """Add query to dataset asynchronously using AsyncClient."""
        global _dataset_id_cache
        try:
            async with ls.AsyncClient() as client:
                # Get or create dataset (cache the ID)
                if _dataset_id_cache is None:
                    try:
                        dataset = await client.read_dataset(
                            dataset_name=GUARDRAILS_DATASET_NAME
                        )
                        _dataset_id_cache = str(dataset.id)
                    except Exception:
                        dataset = await client.create_dataset(
                            dataset_name=GUARDRAILS_DATASET_NAME,
                            description="Production samples for guardrails evaluation. Contains all blocked queries and 10% of allowed queries.",
                        )
                        _dataset_id_cache = str(dataset.id)

                await client.create_example(
                    dataset_id=_dataset_id_cache,
                    inputs={"query": query},
                    outputs={
                        "expected_result": result,
                        "explanation": explanation,
                    },
                )
                logger.debug(f"Added {result} query to dataset: {preview}...")
        except Exception as e:
            logger.warning(f"Failed to add query to dataset: {e}")

    async def _generate_rejection_message(self, content: str) -> AIMessage:
        """Generate a friendly rejection message for off-topic queries."""
        prompt = [
            SystemMessage(content=_REJECTION_SYSTEM_PROMPT),
            HumanMessage(
                content=f"The user asked: {content}\n\nGenerate a brief, friendly response explaining this is outside your scope."
            ),
        ]

        try:
            response = await self.llm.ainvoke(prompt)
            return AIMessage(content=response.content)
        except Exception as e:
            logger.error(f"Error generating rejection message: {e}")
            return AIMessage(content=_FALLBACK_REJECTION_MESSAGE)

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(
        self, state: GuardrailsState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check if query is LangChain-related before processing."""
        messages = state.get("messages", [])
        if not messages:
            return None

        # Extract the current query for all checks below.
        last_message = messages[-1]
        last_content = (
            last_message.content
            if hasattr(last_message, "content")
            else str(last_message)
        )
        query_preview = (
            last_content[:100]
            if isinstance(last_content, str)
            else str(last_content)[:100]
        )

        # One classifier, every turn. Covers topic relevance + zero-tolerance
        # categories (NSFW, fiction, harmful-use-case, prompt-extraction,
        # social-pressure). The prompt's lenient follow-up rules keep legit
        # mid-conversation follow-ups ("show in Python", "3rd one") ALLOWED,
        # while zero-tolerance bullets override the default ALLOW.
        guardrails_decision = await self._classify_query(messages)
        if guardrails_decision is None:
            # Classification failed - allow query through (fail-open)
            return None

        decision = guardrails_decision.decision
        explanation = guardrails_decision.explanation

        # Track in LangSmith metadata
        self._track_decision_metadata(guardrails_decision)

        # Sample to dataset for evaluation (100% blocked, 10% allowed)
        if decision == "BLOCKED" or random.random() < ALLOWED_SAMPLE_RATE:
            asyncio.create_task(
                self._add_to_dataset(last_content, decision, explanation, query_preview)
            )

        # Handle allowed queries
        if decision == "ALLOWED":
            logger.info("Query validated: %s", explanation)
            return None

        # Handle blocked queries
        logger.warning(
            "Off-topic query detected: %s... Reason: %s",
            query_preview,
            explanation,
        )

        if not self.block_off_topic:
            logger.info(
                "Off-topic query detected but block_off_topic=False, allowing..."
            )
            return None

        # Generate rejection and block
        off_topic_message = await self._generate_rejection_message(last_content)
        return {
            "messages": [off_topic_message],
            "off_topic_query": True,
            "jump_to": "end",
        }

    def _extract_message_text(self, msg) -> str | None:
        """Extract plain text content from a message."""
        content = getattr(msg, "content", None)
        if not content:
            return None

        if isinstance(content, str):
            return content.strip() or None

        # Handle list content (extract text blocks only)
        if isinstance(content, list):
            parts = [
                b if isinstance(b, str) else b.get("text", "")
                for b in content
                if isinstance(b, str)
                or (isinstance(b, dict) and b.get("type") == "text")
            ]
            return " ".join(parts).strip() or None

        return None

    async def _classify_query(self, messages: list) -> GuardrailsDecision | None:
        """Classify query as ALLOWED or BLOCKED.

        Returns:
            GuardrailsDecision, or None if classification failed
        """
        # Extract the current query (last human message)
        current_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                current_query = self._extract_message_text(msg)
                if current_query:
                    break

        if not current_query:
            return GuardrailsDecision(
                decision="ALLOWED",
                explanation="No human query was available to classify.",
            )

        # Build context from previous human messages (for follow-up detection)
        prior_queries = []
        for msg in messages[:-1]:  # Exclude current message
            if isinstance(msg, HumanMessage):
                text = self._extract_message_text(msg)
                if text:
                    prior_queries.append(text[:200])  # Truncate for brevity

        # Build the classification prompt
        context_section = ""
        if prior_queries:
            recent = prior_queries[-3:]  # Last 3 prior queries
            context_section = (
                "\n\nPrevious questions in this conversation:\n"
                + "\n".join(f"- {q}" for q in recent)
            )

        prompt = [
            SystemMessage(content=_GUARDRAILS_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Classify this query: {current_query}{context_section}\n\n"
                    "Return both the decision and one concise sentence explaining why."
                )
            ),
        ]

        try:
            structured_llm = self.llm.with_structured_output(GuardrailsDecision)
            result: GuardrailsDecision = await structured_llm.ainvoke(
                prompt, config={"callbacks": [], "tags": ["guardrails"]}
            )
            return result
        except Exception as e:
            logger.error(f"Error in guardrails classification: {e}")
            logger.info("Guardrails check failed, allowing query through...")
            return None

    def _track_decision_metadata(self, decision: GuardrailsDecision) -> None:
        """Add guardrails decision to LangSmith run metadata."""
        try:
            run_tree = ls.get_current_run_tree()
            if run_tree:
                run_tree.metadata["guardrails_result"] = decision.decision
                run_tree.metadata["guardrails_explanation"] = decision.explanation
        except Exception:
            pass  # Silently ignore if run tree is not available


__all__ = ["GuardrailsMiddleware"]
