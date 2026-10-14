"""Enforce documentation link and grounding rules on final responses."""

import re
from typing import Any
from urllib.parse import urlparse

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

NO_RELEVANT_DOCS = "I could not find relevant documentation to answer this question."
BANNED_HOSTS = {"python.langchain.com", "js.langchain.com"}
URL_PATTERN = re.compile(r"https?://(?:[^\s<>()\[\]\"]+|\([^()\s]*\))+", re.IGNORECASE)
LINK_PATTERN = re.compile(
    r"\[([^\]]*)\]\(\s*<?(" + URL_PATTERN.pattern + r")>?(?:\s+[\"'][^)]*[\"'])?\s*\)",
    re.IGNORECASE,
)
DOCS_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?Relevant docs:(?:\*\*)?\s*$", re.IGNORECASE
)
DOCS_CONTEXT = re.compile(
    r"\b(?:docs?|documentation|documented|pages?|guides?)\b", re.IGNORECASE
)
MIGRATION_CLAIM = re.compile(
    r"\b(?:docs?|documentation|pages?|guides?|they|it)\s+"
    r"(?:(?:has|have|had|was|were|is|are|been|being|now|recently|already|all|officially)\s+)*"
    r"(?:moved|migrated|relocated)\b|"
    r"\b(?:moved|migrated|relocated)\s+(?:the\s+)?(?:docs?|documentation|pages?|guides?)\b",
    re.IGNORECASE,
)
LEGACY_REFERENCE = re.compile(
    r"\b(?:legacy|older|old|previous|archived)\s+(?:[\w/-]+\s+){0,2}(?:documentation|docs?|sites?|domains?|pages?|guides?)\b",
    re.IGNORECASE,
)
LEGACY_LOCATION = re.compile(
    r"\b(?:still|hosted|live|lives|remain|available|located|maintained)\b",
    re.IGNORECASE,
)
ABSENCE_CLAIM = re.compile(
    r"\b(?:does not|do not|doesn't|don't|doesn’t|don’t|not|no longer)\s+(?:exist|cover|include)\b",
    re.IGNORECASE,
)
READ_TOOLS = {"query_docs_filesystem_docs_by_lang_chain", "get_support_article_content"}


def _plain_prose(text: str) -> str:
    """Ignore inline Markdown formatting when checking prose claims."""
    return re.sub(r"[*_`]", "", LINK_PATTERN.sub(lambda match: match[1], text))


def _text_content(content: Any) -> str:
    """Extract only text from string or block content."""
    if isinstance(content, str):
        return content
    return "\n".join(
        block if isinstance(block, str) else block.get("text", "")
        for block in content
        if isinstance(block, str)
        or isinstance(block, dict)
        and block.get("type") == "text"
    )


def _tool_evidence(messages: list) -> tuple[set[str], str]:
    """Collect validated URLs and read evidence from tool results, not inputs."""
    tool_names = {
        call["id"]: call["name"]
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
    }
    valid_urls: set[str] = set()
    read_content = []
    for message in messages:
        if not isinstance(message, ToolMessage) or message.status == "error":
            continue
        name = message.name or tool_names.get(message.tool_call_id)
        text = _text_content(message.content)
        if name in READ_TOOLS:
            read_content.append(text)
        if name != "check_links":
            continue
        in_valid_links = False
        for line in text.splitlines():
            if line.strip() == "Valid links:":
                in_valid_links = True
            elif line.strip() == "Invalid links:":
                in_valid_links = False
            elif in_valid_links and line.lstrip().startswith("- "):
                valid_urls.update(URL_PATTERN.findall(line))
            elif line.strip():
                in_valid_links = False
    return valid_urls, " ".join(_plain_prose(" ".join(read_content)).lower().split())


def _sanitize_text(
    text: str, valid_urls: set[str], evidence: str, in_docs: bool
) -> tuple[str, bool]:
    """Remove unsafe links and unsupported documentation-location claims."""
    lines = []
    replaced_claim = False
    for line in text.splitlines(keepends=True):
        if DOCS_HEADING.match(line):
            in_docs = True
        elif line.lstrip().startswith("#"):
            in_docs = False
        urls = URL_PATTERN.findall(line)
        if (
            in_docs
            and re.match(r"\s*(?:[-*+] |\d+[.)] )", line)
            and any(
                url not in valid_urls or urlparse(url).hostname in BANNED_HOSTS
                for url in urls
            )
        ):
            continue
        sentences = re.split(r"(?<=[.!?])(?=\s+[A-Z*])", line)
        for sentence in sentences:
            plain_sentence = _plain_prose(sentence)
            doc_context = bool(DOCS_CONTEXT.search(plain_sentence)) or bool(
                re.match(
                    r"\s*(?:They|It|These|Those|This)\b", plain_sentence, re.IGNORECASE
                )
                and DOCS_CONTEXT.search(_plain_prose(line))
            )
            banned_urls = any(
                urlparse(url).hostname in BANNED_HOSTS
                for url in URL_PATTERN.findall(sentence)
            )
            legacy = bool(LEGACY_REFERENCE.search(plain_sentence))
            legacy = legacy or bool(
                doc_context and banned_urls and LEGACY_LOCATION.search(plain_sentence)
            )
            if re.match(
                r"\s*(?:do not|don't|don’t|never)\s+(?:use|visit|recommend|consult)\b|\s*avoid\b",
                plain_sentence,
                re.IGNORECASE,
            ):
                legacy = False
            unsupported_claim = doc_context and (
                MIGRATION_CLAIM.search(plain_sentence)
                or ABSENCE_CLAIM.search(plain_sentence)
            )
            normalized = " ".join(plain_sentence.lower().split()).strip()
            if legacy or unsupported_claim and normalized not in evidence:
                leading = sentence[: len(sentence) - len(sentence.lstrip())]
                ending = "\n" if sentence.endswith("\n") else ""
                sentence = (
                    "" if replaced_claim else leading + NO_RELEVANT_DOCS
                ) + ending
                replaced_claim = True
            lines.append(sentence)
    cleaned = "".join(lines)
    cleaned = LINK_PATTERN.sub(
        lambda match: (
            match[1] if urlparse(match[2]).hostname in BANNED_HOSTS else match[0]
        ),
        cleaned,
    )
    return cleaned, in_docs


class DocsOutputMiddleware(AgentMiddleware):
    """Guard final AI messages without interfering with tool calls."""

    def after_model(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        """Sanitize the final response while preserving its message identity."""
        messages = state.get("messages", [])
        if (
            not messages
            or not isinstance(messages[-1], AIMessage)
            or messages[-1].tool_calls
        ):
            return None
        message = messages[-1]
        valid_urls, evidence = _tool_evidence(messages)
        content = message.content
        in_docs = False
        cleaned: Any
        if isinstance(content, str):
            cleaned, in_docs = _sanitize_text(content, valid_urls, evidence, in_docs)
            cleaned = re.sub(
                r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*)?Relevant docs:(?:\*\*)?\s*\Z",
                "",
                cleaned,
            )
        else:
            cleaned = []
            for block in content:
                if isinstance(block, str):
                    text, in_docs = _sanitize_text(block, valid_urls, evidence, in_docs)
                    cleaned.append(text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text, in_docs = _sanitize_text(
                        block["text"], valid_urls, evidence, in_docs
                    )
                    cleaned.append({**block, "text": text})
                else:
                    cleaned.append(block)
        if cleaned == content:
            return None
        return {"messages": [message.model_copy(update={"content": cleaned})]}

    async def aafter_model(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        """Apply the same final-response guard during asynchronous execution."""
        return self.after_model(state, runtime)
