"""Retry and sanitize tool-call failures before they reach users."""

import asyncio
import json
import logging
import re
import shlex
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
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


def _normalize_docs_command(command: str) -> str:
    """Repair unambiguous grep-style recursion clusters without rebuilding shell text."""
    if any(char in command for char in "\n\r$`"):
        return command
    token_pattern = re.compile(
        r"""(?:[^\s'"\\|;&<>()#]+|\\[^\n]|'[^']*'|"(?:\\.|[^"\\])*")+|&&|\|\||[|;]"""
    )
    tokens = list(token_pattern.finditer(command))
    end = 0
    for token in tokens:
        if command[end : token.start()].strip():
            return command
        end = token.end()
    if command[end:].strip():
        return command

    replacements = []
    segments: list[list[re.Match[str]]] = [[]]
    for token in tokens:
        if token.group() in {"|", ";", "&&", "||"}:
            segments.append([])
        else:
            segments[-1].append(token)
    for segment in segments:
        if not segment or segment[0].group() != "rg":
            continue
        try:
            words = [shlex.split(token.group())[0] for token in segment]
        except (ValueError, IndexError):
            return command
        if any(word == "-r" or word.split("=", 1)[0] == "--replace" for word in words):
            continue
        pending = []
        index = 1
        while index < len(words):
            word = words[index]
            if word == "--" or not word.startswith("-") or word == "-":
                break
            if (
                re.fullmatch(r"-[nilr]+", word)
                and word.count("r") == 1
                and len(word) > 2
            ):
                token = segment[index]
                if token.group() == word:
                    pending.append((token.start(), token.end(), word.replace("r", "")))
            elif re.fullmatch(r"-[nil]+", word):
                pass
            elif word in {
                "-A",
                "-B",
                "-C",
                "-e",
                "-f",
                "-g",
                "-t",
                "-T",
                "-m",
                "--after-context",
                "--before-context",
                "--context",
                "--regexp",
                "--file",
                "--glob",
                "--iglob",
                "--type",
                "--type-not",
                "--max-count",
                "--encoding",
            }:
                index += 1
                if index >= len(words):
                    pending = []
                    break
            else:
                pending = []
                break
            index += 1
        replacements.extend(pending)
    for start, end, replacement in reversed(replacements):
        command = command[:start] + replacement + command[end:]
    return command


# Docs MCP tools whose rare stalled calls are worth hedging. Prod (Nov 17-23):
# healthy search p99 ~3s, filesystem p99 ~5s, but ~1-2% of traces hit a call
# that stalls to the 10s client timeout, and the same call retried succeeds
# ~85-95% of the time. Starting one duplicate after HEDGE_AFTER_SECONDS rescues
# those stalls ~5s earlier without failing legitimately slow commands.
HEDGED_TOOLS = frozenset(
    {
        "langchainDocs_search_docs_by_lang_chain",
        "langchainDocs_query_docs_filesystem_docs_by_lang_chain",
    }
)
HEDGE_AFTER_SECONDS = 5.0


class ToolRetryMiddleware(AgentMiddleware[AgentState]):
    """Retry transient tool failures and return model-readable errors."""

    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 0.5,
        backoff_factor: float = 2.0,
        hedge_after: float | None = HEDGE_AFTER_SECONDS,
    ):
        super().__init__()
        self.hedge_after = hedge_after
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

    async def _call(self, request: ToolCallRequest, handler):
        """Run one attempt; for hedged tools, race a duplicate after a delay."""
        if self.hedge_after is None or self._tool_name(request) not in HEDGED_TOOLS:
            return await handler(request)

        first = asyncio.ensure_future(handler(request))
        second: asyncio.Future | None = None
        try:
            done, _ = await asyncio.wait({first}, timeout=self.hedge_after)
            if done:
                return first.result()

            logger.info(
                "Tool %s still running after %.1fs; starting hedged duplicate",
                self._tool_name(request),
                self.hedge_after,
            )
            second = asyncio.ensure_future(handler(request))
            pending = {first, second}
            error: BaseException | None = None
            while pending:
                done, pending = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task.exception() is None:
                        return task.result()
                    error = error or task.exception()
            assert error is not None
            raise error
        finally:
            for task in (first, second):
                if task is not None and not task.done():
                    task.cancel()

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        if (
            self._tool_name(request)
            == "langchainDocs_query_docs_filesystem_docs_by_lang_chain"
        ):
            args = request.tool_call.get("args", {})
            command = args.get("command")
            if isinstance(command, str):
                normalized = _normalize_docs_command(command)
                if normalized != command:
                    logger.info("Normalized docs filesystem rg recursion flags")
                    request = request.override(
                        tool_call={
                            **request.tool_call,
                            "args": {**args, "command": normalized},
                        }
                    )

        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                return await self._call(request, handler)
            except Exception as error:
                last_error = error
                tool_name = self._tool_name(request)

                if self._is_no_results(error):
                    logger.info(
                        "Tool %s returned no results; normalizing as tool output",
                        tool_name,
                    )
                    return self._tool_message(request, "No results found.")

                if (
                    tool_name
                    == "langchainDocs_query_docs_filesystem_docs_by_lang_chain"
                    and any(
                        marker in self._error_text(error).lower()
                        for marker in ("timed out", "timeout")
                    )
                    and attempt >= min(2, self.max_attempts)
                ):
                    return self._tool_message(
                        request,
                        json.dumps(
                            {
                                "error": "Docs command timed out",
                                "message": (
                                    "The command exceeded the docs server's "
                                    "10-second limit."
                                ),
                                "command": request.tool_call["args"].get("command", "")[
                                    :120
                                ],
                                "tool": tool_name,
                                "suggestion": (
                                    "Scope rg to a specific folder or file, or use "
                                    "langchainDocs_search_docs_by_lang_chain."
                                ),
                            }
                        ),
                    )

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
