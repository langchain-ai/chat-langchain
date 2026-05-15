# Lenient guardrails middleware to filter only egregious misuse
import asyncio
import logging
import os
import random
from typing import Any, Literal

import langsmith as ls
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from langsmith import Client
from pydantic import BaseModel, Field
from typing_extensions import NotRequired

from src.prompts.guardrails_prompts import (
    fallback_rejection_message as _FALLBACK_REJECTION_MESSAGE,
)
from src.prompts.guardrails_prompts import (
    guardrails_system_prompt as _LOCAL_GUARDRAILS_SYSTEM_PROMPT,
)
from src.prompts.guardrails_prompts import (
    rejection_system_prompt as _REJECTION_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

# Dataset configuration for guardrails evaluation
GUARDRAILS_DATASET_NAME = "Chat-LangChain-Guardrails-Samples"
ALLOWED_SAMPLE_RATE = 0.01  # 1% of allowed queries go to dataset
GUARDRAILS_MAX_RETRIES = 2
GUARDRAILS_TIMEOUT_SECONDS = 10
GUARDRAILS_FAILURE_MESSAGE = "something went wrong, please try again"
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


class GuardrailsClassificationError(Exception):
    """Raised when guardrails classification fails after retries."""

    pass


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

    async def _generate_rejection_message(self, content) -> AIMessage:
        """Generate a friendly rejection message for off-topic queries."""
        prompt = [
            SystemMessage(content=_REJECTION_SYSTEM_PROMPT),
            HumanMessage(
                content=self._build_rejection_content(content)
            ),
        ]

        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(prompt),
                timeout=GUARDRAILS_TIMEOUT_SECONDS,
            )
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
        safe_last_content = self._content_to_safe_text(last_content)
        query_preview = safe_last_content[:100]

        # One classifier, every turn. Covers topic relevance + zero-tolerance
        # categories (NSFW, fiction, harmful-use-case, prompt-extraction,
        # social-pressure). The prompt's lenient follow-up rules keep legit
        # mid-conversation follow-ups ("show in Python", "3rd one") ALLOWED,
        # while zero-tolerance bullets override the default ALLOW.
        try:
            guardrails_decision = await self._classify_query(messages)
        except GuardrailsClassificationError:
            logger.error("Guardrails check failed after retries; stopping run.")
            return {
                "messages": [AIMessage(content=GUARDRAILS_FAILURE_MESSAGE)],
                "off_topic_query": False,
                "jump_to": "end",
            }

        decision = guardrails_decision.decision
        explanation = guardrails_decision.explanation

        # Track in LangSmith metadata
        self._track_decision_metadata(guardrails_decision)

        # Sample to dataset for evaluation (100% blocked, 10% allowed)
        if decision == "BLOCKED" or random.random() < ALLOWED_SAMPLE_RATE:
            asyncio.create_task(
                self._add_to_dataset(
                    safe_last_content,
                    decision,
                    explanation,
                    query_preview,
                )
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

    def _content_to_safe_text(self, content) -> str:
        """Convert multimodal content to text without leaking encoded media."""
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return str(content)

        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue

            if not isinstance(block, dict):
                continue

            block_type = block.get("type")
            if block_type == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
            elif block_type in ("image_url", "input_image"):
                parts.append("[image attached]")
            elif block_type:
                parts.append(f"[{block_type} attached]")

        return " ".join(part for part in parts if part).strip()

    def _build_rejection_content(self, content) -> str | list:
        """Build rejection prompt while preserving images for vision-capable models."""
        instruction = (
            "The user asked the following request. Consider both the text and any "
            "attached images, then generate a brief, friendly response explaining "
            "this is outside your scope."
        )

        if not isinstance(content, list):
            return f"{instruction}\n\nUser request: {content}"

        blocks: list[Any] = [{"type": "text", "text": instruction}]
        for block in content:
            if isinstance(block, str):
                blocks.append({"type": "text", "text": block})
            elif isinstance(block, dict):
                blocks.append(block)

        blocks.append(
            {
                "type": "text",
                "text": (
                    "Generate a brief, friendly response explaining this is outside "
                    "your scope."
                ),
            }
        )
        return blocks

    def _content_has_media(self, content) -> bool:
        """Return whether content contains non-text multimodal blocks."""
        if not isinstance(content, list):
            return False

        return any(
            isinstance(block, dict) and block.get("type") not in (None, "text")
            for block in content
        )

    def _build_guardrails_content(self, content, context_section: str) -> str | list:
        """Build classifier input while preserving image blocks for vision models."""
        instruction = (
            "Classify this user query for the LangChain documentation assistant. "
            "Consider both the text and any attached images. "
            "Return both the decision and one concise sentence explaining why."
        )

        if context_section:
            instruction += context_section

        if not isinstance(content, list):
            return f"{instruction}\n\nUser query: {content}"

        blocks: list[Any] = [{"type": "text", "text": instruction}]
        for block in content:
            if isinstance(block, str):
                blocks.append({"type": "text", "text": block})
            elif isinstance(block, dict):
                blocks.append(block)

        return blocks

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

    async def _classify_query(self, messages: list) -> GuardrailsDecision:
        """Classify query as ALLOWED or BLOCKED.

        Raises:
            GuardrailsClassificationError: If classification fails after retries.
        """
        # Extract the current query (last human message)
        current_message = None
        current_query = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                current_message = msg
                current_query = self._extract_message_text(msg)
                if current_query or self._content_has_media(getattr(msg, "content", None)):
                    break

        if current_message is None or (
            not current_query
            and not self._content_has_media(getattr(current_message, "content", None))
        ):
            return GuardrailsDecision(
                decision="ALLOWED",
                explanation="No human query was available to classify.",
            )

        # Build context from previous human messages (for follow-up detection)
        prior_queries = []
        for msg in reversed(messages[:-1]):  # Exclude current message
            if isinstance(msg, HumanMessage):
                text = self._extract_message_text(msg)
                if text:
                    prior_queries.append(text[:200])  # Truncate for brevity
                    if len(prior_queries) == 3:
                        break

        # Build the classification prompt
        context_section = ""
        if prior_queries:
            recent = list(reversed(prior_queries))  # Restore chronological order.
            context_section = (
                "\n\nPrevious questions in this conversation:\n"
                + "\n".join(f"- {q}" for q in recent)
            )

        current_content = getattr(current_message, "content", current_query or "")
        prompt = [
            SystemMessage(content=_GUARDRAILS_SYSTEM_PROMPT),
            HumanMessage(
                content=self._build_guardrails_content(current_content, context_section)
            ),
        ]

        structured_llm = self.llm.with_structured_output(GuardrailsDecision)
        last_exception: Exception | None = None

        for attempt in range(GUARDRAILS_MAX_RETRIES + 1):
            try:
                result: GuardrailsDecision = await asyncio.wait_for(
                    structured_llm.ainvoke(
                        prompt, config={"callbacks": [], "tags": ["guardrails"]}
                    ),
                    timeout=GUARDRAILS_TIMEOUT_SECONDS,
                )
                return result
            except Exception as e:
                last_exception = e
                if attempt < GUARDRAILS_MAX_RETRIES:
                    logger.warning(
                        "Guardrails classification failed attempt %s/%s: %s. Retrying...",
                        attempt + 1,
                        GUARDRAILS_MAX_RETRIES + 1,
                        e,
                    )
                    continue

                logger.error(
                    "Guardrails classification failed after %s attempts: %s",
                    GUARDRAILS_MAX_RETRIES + 1,
                    e,
                )

        raise GuardrailsClassificationError(
            f"Guardrails classification failed after retries: {last_exception}"
        )

    def _track_decision_metadata(self, decision: GuardrailsDecision) -> None:
        """Add guardrails decision to LangSmith run metadata."""
        try:
            run_tree = ls.get_current_run_tree()
            if run_tree:
                run_tree.metadata["guardrails_result"] = decision.decision
                run_tree.metadata["guardrails_explanation"] = decision.explanation
        except Exception:
            pass  # Silently ignore if run tree is not available


__all__ = ["GuardrailsMiddleware", "GuardrailsClassificationError"]
