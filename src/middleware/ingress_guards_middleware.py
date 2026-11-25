"""Ingress guards for Chat LangChain on Managed Deep Agents.

These were previously enforced in ``src/api/auth.py`` (``validate_inputs``).
Under MDA, identity/thread scoping is declared in ``identity.py``; this
middleware handles content-free greetings, caps input, and redacts credentials.

Trace metadata (prompt provenance, ``LANGSMITH_AGENT_VERSION``, ``source_type``)
is applied at agent compile time via ``define_deep_agent(metadata=...)`` in
``agent.py``. Credential redaction metadata is applied through the current run tree,
with root-linked feedback when only parent IDs are available. Git-linked host
fields (``LANGSMITH_LANGGRAPH_GIT_*``) are not synthesized; archive deploys use ``LANGSMITH_HOST_REVISION_ID`` /
``LANGSMITH_AGENT_VERSION`` instead.
"""

from __future__ import annotations

import re
from typing import Any

import langsmith as ls
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

#: Upper bound on user-provided text, matching the previous ``MAX_MESSAGE_CHARS``.
MAX_MESSAGE_CHARS = 50_000


_PREFIXED_SECRET = (
    r"\b(?:sk-(?:ant-)?|tvly-|AIza|gh[po]_|xoxb-|pk_live_|lsv2_|lcl_)[A-Za-z0-9_-]+"
)
# The generic token branch is bounded (512 chars): unbounded lookaheads rescan the
# rest of the message at every "key=" position, which is quadratic and froze the
# event loop for ~8s on a 50k-char "key=key=..." message (an easy DoS).
_SECRET_RE = re.compile(
    r"(?P<label>\b[\w-]*(?:key|token|secret|password)[\"']?\s*[:=]\s*)"
    r"(?P<quote>[\"']?)(?:"
    + _PREFIXED_SECRET
    + r"|(?=[A-Za-z0-9_\-.+=]{0,511}[A-Za-z])(?=[A-Za-z0-9_\-.+=]{0,511}[0-9])"
    r"[A-Za-z0-9_\-.+=]{16,512})(?P=quote)(?![A-Za-z0-9_\-.+=/('\"\[])"
    r"|" + _PREFIXED_SECRET + r"|\bBearer[ \t]+[A-Za-z0-9._~+/=-]+"
    r"|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+",
    re.IGNORECASE,
)


_GREETING_REPLIES = [
    (
        ("hi", "hello", "hey", "thanks", "thank you", "ok", "okay"),
        "Hi! How can I help with LangChain, LangGraph, LangSmith, or Deep Agents?",
    ),
    (
        ("你好", "您好", "嗨", "谢谢", "謝謝", "好的"),
        "你好！有什么关于 LangChain、LangGraph、LangSmith 或 Deep Agents 的问题需要帮助吗？",
    ),
    (
        ("こんにちは", "ありがとう", "ありがとうございます", "はい"),
        "こんにちは！LangChain、LangGraph、LangSmith、Deep Agents について、どのようなご質問がありますか？",
    ),
    (
        ("안녕하세요", "안녕", "감사합니다", "고마워", "네"),
        "안녕하세요! LangChain, LangGraph, LangSmith 또는 Deep Agents에 대해 무엇을 도와드릴까요?",
    ),
    (
        ("hola", "gracias", "vale"),
        "¡Hola! ¿En qué puedo ayudarte con LangChain, LangGraph, LangSmith o Deep Agents?",
    ),
    (
        ("bonjour", "salut", "merci", "d’accord", "d'accord"),
        "Bonjour ! Comment puis-je vous aider avec LangChain, LangGraph, LangSmith ou Deep Agents ?",
    ),
    (
        ("hallo", "danke", "danke schön"),
        "Hallo! Wie kann ich bei Fragen zu LangChain, LangGraph, LangSmith oder Deep Agents helfen?",
    ),
    (
        ("olá", "oi", "obrigado", "obrigada"),
        "Olá! Como posso ajudar com LangChain, LangGraph, LangSmith ou Deep Agents?",
    ),
    (
        ("ciao", "grazie"),
        "Ciao! Come posso aiutarti con LangChain, LangGraph, LangSmith o Deep Agents?",
    ),
    (
        ("привет", "здравствуйте", "спасибо", "хорошо"),
        "Здравствуйте! Чем могу помочь с LangChain, LangGraph, LangSmith или Deep Agents?",
    ),
    (
        ("مرحبا", "مرحباً", "أهلا", "شكرا", "شكراً"),
        "مرحباً! كيف يمكنني مساعدتك في أسئلة حول LangChain أو LangGraph أو LangSmith أو Deep Agents؟",
    ),
    (
        ("नमस्ते", "धन्यवाद"),
        "नमस्ते! LangChain, LangGraph, LangSmith या Deep Agents के बारे में मैं आपकी क्या मदद कर सकता हूँ?",
    ),
]
_GREETING_LOOKUP = {
    greeting: reply for greetings, reply in _GREETING_REPLIES for greeting in greetings
}


def _greeting_reply(content: Any) -> str | None:
    """Match an entire text-only turn, excluding the appended page note."""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                return None
        content = " ".join(parts)
    if not isinstance(content, str):
        return None
    body = content.partition("\n---\nNote: The user is asking")[0]
    greeting = " ".join(body.casefold().split()).strip(" !.,?¡¿。！？،؛")
    return _GREETING_LOOKUP.get(greeting)


class IngressGuardsMiddleware(AgentMiddleware):
    """Handle greetings, cap input, and redact credentials at agent ingress."""

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Handle content-free greetings, then cap and redact user input."""
        messages = state.get("messages", [])
        humans = [message for message in messages if isinstance(message, HumanMessage)]
        if humans and all(_greeting_reply(message.content) for message in humans[:-1]):
            reply = _greeting_reply(humans[-1].content)
            if reply:
                return {"messages": [AIMessage(content=reply)], "jump_to": "end"}
        for message in reversed(messages):
            if getattr(message, "type", None) == "human":
                capped = self._truncate_content(message.content)
                redacted = self._redact_content(capped)
                if redacted != capped:
                    self._track_redaction_metadata()
                if capped is not message.content or redacted != capped:
                    # Same id => the messages reducer overwrites in place.
                    message.content = redacted
                    return {"messages": [message]}
                break
        return None

    def _redact_content(self, content: Any) -> Any:
        """Replace credentials in text while preserving non-text blocks."""
        if isinstance(content, str):
            return _SECRET_RE.sub(
                lambda match: (match.group("label") or "") + "[REDACTED_SECRET]",
                content,
            )
        if not isinstance(content, list):
            return content
        return [
            self._redact_content(block)
            if isinstance(block, str)
            else {**block, "text": self._redact_content(block["text"])}
            if isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            else block
            for block in content
        ]

    def _track_redaction_metadata(self) -> None:
        """Mark runs whose latest user message contained credentials."""
        try:
            run_tree = ls.get_current_run_tree()
            if run_tree:
                while run_tree.parent_run is not None:
                    run_tree = run_tree.parent_run
                run_tree.metadata["secret_redacted"] = True
                # Some SDK versions retain parent IDs without parent objects.
                if run_tree.parent_run_id is not None:
                    run_tree.client.create_feedback(
                        run_id=run_tree.trace_id, key="secret_redacted", score=1
                    )
        except Exception:
            pass

    def _truncate_content(self, content: Any) -> Any:
        """Trim user text to the cap while preserving non-text content blocks."""
        if isinstance(content, str):
            return (
                content[:MAX_MESSAGE_CHARS]
                if len(content) > MAX_MESSAGE_CHARS
                else content
            )

        if not isinstance(content, list):
            return content

        remaining = MAX_MESSAGE_CHARS
        changed = False
        truncated: list[Any] = []
        for block in content:
            if isinstance(block, str):
                text = block[:remaining]
                changed = changed or len(text) != len(block)
                truncated.append(text)
                remaining -= len(text)
            elif (
                isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ):
                text = block["text"][:remaining]
                changed = changed or len(text) != len(block["text"])
                truncated.append({**block, "text": text})
                remaining -= len(text)
            else:
                truncated.append(block)
        return truncated if changed else content


__all__ = ["IngressGuardsMiddleware", "MAX_MESSAGE_CHARS"]
