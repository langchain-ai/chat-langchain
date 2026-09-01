"""Remove unvalidated official documentation links from model responses."""

from __future__ import annotations

import re
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState

_OFFICIAL_DOCS_HOSTS = ("docs.langchain.com", "reference.langchain.com")
_URL_PATTERN = re.compile(r"https?://[^\s<>()\[\]{}]+")
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


class DocsLinkGuardMiddleware(AgentMiddleware):
    """Strip official documentation URLs not validated during the current turn."""

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        """Remove unvalidated official documentation links from the latest answer."""
        messages = state.get("messages", [])
        if not messages:
            return None

        latest_human_index = max(
            (index for index, message in enumerate(messages) if self._message_type(message) == "human"),
            default=-1,
        )
        valid_urls = self._valid_urls(messages[latest_human_index + 1 :])
        answer = messages[-1]
        if self._message_type(answer) != "ai" or getattr(answer, "tool_calls", None):
            return None

        sanitized_content = self._sanitize_content(getattr(answer, "content", ""), valid_urls)
        if sanitized_content == answer.content:
            return None
        answer.content = sanitized_content
        return {"messages": [answer]}

    @staticmethod
    def _message_type(message: Any) -> str | None:
        return getattr(message, "type", None) or getattr(message, "role", None)

    @classmethod
    def _valid_urls(cls, messages: list[Any]) -> set[str]:
        valid_urls: set[str] = set()
        for message in messages:
            if cls._message_type(message) != "tool" or getattr(message, "name", None) != "check_links":
                continue
            content = cls._content_to_text(getattr(message, "content", ""))
            match = re.search(r"Valid links:\s*(.*)", content, re.DOTALL)
            if match:
                valid_urls.update(cls._normalize_url(url) for url in _URL_PATTERN.findall(match.group(1)))
        return valid_urls

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                block if isinstance(block, str) else str(block.get("text", ""))
                for block in content
                if isinstance(block, (str, dict))
            )
        return str(content)

    @classmethod
    def _sanitize_content(cls, content: Any, valid_urls: set[str]) -> Any:
        if isinstance(content, str):
            return cls._sanitize_text(content, valid_urls)
        if not isinstance(content, list):
            return content

        sanitized = []
        for block in content:
            if isinstance(block, str):
                sanitized.append(cls._sanitize_text(block, valid_urls))
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                sanitized.append({**block, "text": cls._sanitize_text(block["text"], valid_urls)})
            else:
                sanitized.append(block)
        return sanitized

    @classmethod
    def _sanitize_text(cls, text: str, valid_urls: set[str]) -> str:
        lines = text.splitlines(keepends=True)
        sanitized_lines: list[str] = []
        in_docs_section = False
        docs_section_has_link = False

        for line in lines:
            if re.match(r"^\s*\*\*Relevant docs:\*\*\s*$", line.strip()):
                in_docs_section = True
                docs_section_has_link = False
                sanitized_lines.append(line)
                continue

            line_urls = _URL_PATTERN.findall(line)
            official_urls = [url for url in line_urls if cls._is_official_docs_url(url)]
            invalid_official_urls = [url for url in official_urls if cls._normalize_url(url) not in valid_urls]
            if official_urls and not invalid_official_urls:
                docs_section_has_link = docs_section_has_link or in_docs_section

            if invalid_official_urls and in_docs_section and line.lstrip().startswith(("-", "*")):
                continue

            def replace_link(match: re.Match[str]) -> str:
                url = cls._normalize_url(match.group(2))
                return match.group(0) if url in valid_urls else match.group(1)

            line = _MARKDOWN_LINK_PATTERN.sub(replace_link, line)
            for url in invalid_official_urls:
                line = line.replace(url, "")
            sanitized_lines.append(line)

        if in_docs_section and not docs_section_has_link:
            sanitized_lines = [
                line
                for line in sanitized_lines
                if not re.match(r"^\s*\*\*Relevant docs:\*\*\s*$", line.strip())
            ]
        return "".join(sanitized_lines)

    @staticmethod
    def _is_official_docs_url(url: str) -> bool:
        return any(re.match(rf"^https?://{re.escape(host)}(?:/|$)", url, re.IGNORECASE) for host in _OFFICIAL_DOCS_HOSTS)

    @staticmethod
    def _normalize_url(url: str) -> str:
        return url.rstrip(".,;:!?)]}")


__all__ = ["DocsLinkGuardMiddleware"]
