"""Retry and sanitize tool-call failures before they reach users."""

import asyncio
import json
import logging
import re
import shlex
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)

NO_RESULTS_MARKERS = (
    "no results found",
    "no result found",
)

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

RETRYABLE_ERROR_MARKERS = (
    "bad gateway",
    "connection error",
    "connection reset",
    "gateway time-out",
    "gateway timeout",
    "service unavailable",
    "temporarily unavailable",
    "timed out",
    "timeout",
    "too many requests",
)


class ToolRetryMiddleware(AgentMiddleware[AgentState]):
    """Retry transient tool failures and return model-readable errors."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
    ):
        """Configure transient tool retries."""
        super().__init__()
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def _tool_name(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("name", "unknown_tool")

    def _tool_call_id(self, request: ToolCallRequest) -> str:
        return request.tool_call.get("id", "")

    def _error_text(self, error: Exception) -> str:
        return str(error) or error.__class__.__name__

    def _status_code(self, error: Exception) -> int | None:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(error, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int):
            return response_status

        text = self._error_text(error)
        status_match = re.search(
            r"\b(?:HTTP|status(?:\s+code)?|error\s+code)[:= ]+"
            r"(429|500|502|503|504)\b",
            text,
            re.IGNORECASE,
        )
        if status_match:
            return int(status_match.group(1))

        return None

    def _is_no_results(self, error: Exception) -> bool:
        text = self._error_text(error).lower()
        return any(marker in text for marker in NO_RESULTS_MARKERS)

    def _is_retryable(self, error: Exception) -> bool:
        text = self._error_text(error).lower()
        status_code = self._status_code(error)
        if status_code in RETRYABLE_STATUS_CODES:
            return True

        return any(marker in text for marker in RETRYABLE_ERROR_MARKERS)

    def _tool_message(
        self,
        request: ToolCallRequest,
        content: str,
    ) -> ToolMessage:
        return ToolMessage(
            content=content,
            name=self._tool_name(request),
            tool_call_id=self._tool_call_id(request),
        )

    def _final_error_content(
        self,
        request: ToolCallRequest,
        error: Exception,
    ) -> str:
        tool_name = self._tool_name(request)
        payload: dict[str, Any] = {
            "error": "Tool unavailable",
            "message": f"{tool_name} failed after {self.max_attempts} attempts.",
            "tool": tool_name,
            "suggestion": (
                "Try a narrower or related query, use another available source, "
                "or answer from already retrieved context."
            ),
            "details": self._error_text(error)[:160],
        }
        return json.dumps(payload)

    def _docs_read_range(
        self, request: ToolCallRequest
    ) -> tuple[str, int, int | None] | None:
        if (
            self._tool_name(request)
            != "langchainDocs_query_docs_filesystem_docs_by_lang_chain"
        ):
            return None
        command = request.tool_call.get("args", {}).get("command", "")
        if not isinstance(command, str):
            return None
        try:
            parts = shlex.split(command)
        except ValueError:
            return None
        if not parts or not re.fullmatch(r"[\w/.-]+\.mdx", parts[-1]):
            return None
        path = parts[-1]
        if parts[:-1] == ["cat"]:
            return path, 1, None
        if parts[0] == "head":
            count = None
            if len(parts) == 3 and re.fullmatch(r"-\d+", parts[1]):
                count = int(parts[1][1:])
            elif len(parts) == 4 and parts[1] == "-n" and parts[2].isdigit():
                count = int(parts[2])
            if count:
                return path, 1, count
        if len(parts) == 4 and parts[:2] == ["sed", "-n"]:
            match = re.fullmatch(r"([1-9]\d*),([1-9]\d*)p", parts[2])
            if match and int(match[1]) <= int(match[2]):
                return path, int(match[1]), int(match[2])
        if len(parts) == 4 and parts[:2] == ["tail", "-n"]:
            if re.fullmatch(r"\+[1-9]\d*", parts[2]):
                return path, int(parts[2][1:]), None
        return None

    def _docs_output(self, result) -> tuple[str, bool]:
        if not isinstance(result, ToolMessage) or result.status == "error":
            raise ValueError("Docs read did not return a successful tool message")
        content = result.content
        if isinstance(content, list):
            texts = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    raise ValueError("Unexpected docs content blocks")
                texts.append(block["text"])
            text = "".join(texts)
        else:
            text = content
        header = re.match(r"exit: (\d+)\n--- stdout ---\n", text, re.IGNORECASE)
        if header:
            if header[1] != "0":
                raise ValueError("Docs command failed")
            text = text[header.end() :]
        marker = re.search(r"\n?\.\.\. \(output truncated\)\s*$", text)
        if marker:
            text = text[: marker.start()]
            # Only newline-terminated source lines can safely advance the offset.
            text = text[: text.rfind("\n") + 1]
        return text, marker is not None

    def _whole_page_request(self, state) -> bool:
        for message in reversed(state.get("messages", [])):
            if isinstance(message, HumanMessage):
                text = message.text.lower()
                return bool(
                    re.search(r"\b(?:rewrit\w*|translat\w*|summari\w*|summary)\b", text)
                    and (
                        re.search(r"\b(?:whole|entire|full) (?:page|document)\b", text)
                        or not re.search(
                            r"\b(?:section|subsection|paragraph|snippet|introduction)\b",
                            text,
                        )
                    )
                )
        return False

    async def aafter_model(self, state, runtime):
        """Disclose unread page ranges even if the model omits the warning."""
        messages = state.get("messages", [])
        if not messages or not self._whole_page_request(state):
            return None
        answer = messages[-1]
        if not isinstance(answer, AIMessage) or answer.tool_calls:
            return None
        coverage = {}
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            record = message.additional_kwargs.get("docs_coverage")
            if record and record["path"] not in coverage:
                coverage[record["path"]] = record["unread"]
        notices = [
            f"Incomplete documentation coverage: {path}, unread lines {unread}. "
            "This answer does not cover the entire page."
            for path, unread in coverage.items()
            if unread
        ]
        if not notices:
            return None
        notice = "\n\n" + "\n".join(notices)
        content = answer.content
        updated = (
            content + notice
            if isinstance(content, str)
            else [*content, {"type": "text", "text": notice}]
        )
        return {"messages": [answer.model_copy(update={"content": updated})]}

    async def _recover_docs_read(self, request, handler, result):
        read_range = self._docs_read_range(request)
        if read_range is None or not isinstance(result, ToolMessage):
            return result
        path, start, endpoint = read_range
        whole_page = self._whole_page_request(request.state)
        try:
            text, truncated = self._docs_output(result)
        except (ValueError, KeyError, TypeError):
            if not whole_page:
                return result
            supplement = SystemMessage(
                content=f"Documentation coverage unavailable for {path}; unread lines 1-EOF.",
                additional_kwargs={"docs_coverage": {"path": path, "unread": "1-EOF"}},
            )
            return Command(update={"messages": [result, supplement]})
        if not truncated and (not whole_page or (start == 1 and endpoint is None)):
            return result
        if whole_page:
            endpoint = None
            if start != 1:
                start, text = 1, ""
        chunks = [text]
        start += text.count("\n") + int(bool(text) and not text.endswith("\n"))
        unread = None
        size = 200
        try:
            for _ in range(1000):
                if endpoint is not None and start > endpoint:
                    break
                end = min(start + size - 1, endpoint) if endpoint else start + size - 1
                call = {
                    **request.tool_call,
                    "args": {
                        **request.tool_call["args"],
                        "command": f"sed -n '{start},{end}p' {shlex.quote(path)}",
                    },
                }
                continuation = await handler(request.override(tool_call=call))
                chunk, cut = self._docs_output(continuation)
                if cut:
                    if size == 1:
                        raise ValueError(
                            "A single source line exceeds the tool output limit"
                        )
                    size = max(1, size // 2)
                    continue
                chunks.append(chunk)
                lines = chunk.count("\n") + int(
                    bool(chunk) and not chunk.endswith("\n")
                )
                requested_lines = end - start + 1
                start += lines
                if lines < requested_lines:
                    break
            else:
                raise ValueError("Docs recovery read budget exhausted")
        except Exception as error:
            remaining = f"{start}-{endpoint}" if endpoint else f"{start}-EOF"
            unread = remaining
            chunks.append(
                f"\nIncomplete documentation coverage: {path}, unread lines {remaining}. "
                f"Recovery failed: {self._error_text(error)}. "
                "Explicitly disclose this uncovered range in any whole-page answer."
            )
        record = {"path": path, "unread": unread}
        if whole_page and not truncated:
            supplement = SystemMessage(
                content=f"Documentation coverage for {path}:\n" + "".join(chunks),
                additional_kwargs={"docs_coverage": record},
            )
            return Command(update={"messages": [result, supplement]})
        return result.model_copy(
            update={
                "content": "".join(chunks),
                "additional_kwargs": {
                    **result.additional_kwargs,
                    "docs_coverage": record,
                },
            }
        )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        """Retry failures and recover truncated single-page docs reads."""
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                result = await handler(request)
                return await self._recover_docs_read(request, handler, result)
            except Exception as error:
                last_error = error
                tool_name = self._tool_name(request)

                if self._is_no_results(error):
                    logger.info(
                        "Tool %s returned no results; normalizing as tool output",
                        tool_name,
                    )
                    return self._tool_message(request, "No results found.")

                if self._is_retryable(error) and attempt < self.max_attempts:
                    delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
                    logger.warning(
                        "Tool %s failed attempt %s/%s: %s; retrying in %.2fs",
                        tool_name,
                        attempt,
                        self.max_attempts,
                        self._error_text(error),
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                logger.warning(
                    "Tool %s failed after %s/%s attempts: %s",
                    tool_name,
                    attempt,
                    self.max_attempts,
                    self._error_text(error),
                )
                return self._tool_message(
                    request,
                    self._final_error_content(request, error),
                )

        # Defensive fallback; loop should always return on success or final error.
        assert last_error is not None
        return self._tool_message(
            request, self._final_error_content(request, last_error)
        )


__all__ = ["ToolRetryMiddleware"]
