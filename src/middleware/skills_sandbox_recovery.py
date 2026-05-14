"""Fail-open recovery for LangChain skills sandbox initialization.

Agent Builder can inject LangChain's SkillsMiddleware at runtime. Some runs reach
that middleware with a stale thread-scoped sandbox id, causing skill listing to
raise a sandbox 404 before the model has a chance to answer. This module applies
an intentionally narrow compatibility patch: on that specific missing-sandbox
failure, ask any available sandbox handle to recreate/reattach, retry skill
initialization once, and then continue without skills if recovery still fails.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

_F = TypeVar("_F", bound=Callable[..., Any])
_PATCHED_ATTR = "_chat_langchain_missing_sandbox_recovery_patched"

_MISSING_SANDBOX_MARKERS = (
    "sandbox not found",
    "resource not found",
    "resource_not_found",
)

_RECOVERY_METHOD_NAMES = (
    "recreate",
    "arecreate",
    "reattach",
    "areattach",
    "reset",
    "areset",
    "initialize",
    "ainitialize",
    "ensure",
    "aensure",
    "setup",
    "asetup",
    "create",
    "acreate",
)

_SANDBOX_ATTR_NAMES = (
    "sandbox",
    "sandbox_backend",
    "sandbox_client",
    "filesystem",
    "fs",
)


def _is_missing_sandbox_error(error: BaseException) -> bool:
    """Return True for sandbox 404 / ResourceNotFound style errors."""

    status_code = getattr(error, "status_code", None)
    if status_code is None:
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if status_code == 404:
        return True

    error_name = error.__class__.__name__.lower()
    if "resourcenotfound" in error_name or "notfound" in error_name:
        return True

    message = str(error).lower()
    return any(marker in message for marker in _MISSING_SANDBOX_MARKERS)


def _call_with_supported_args(method: Callable[..., Any], *candidates: Any) -> Any:
    """Call a recovery hook, passing runtime/state only when accepted."""

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return method()

    parameters = list(signature.parameters.values())
    required_positional = [
        parameter
        for parameter in parameters
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    ]

    if any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
    ):
        return method(*candidates)

    return method(*candidates[: len(required_positional)])


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _iter_recovery_targets(middleware: Any, runtime: Any | None) -> list[Any]:
    """Collect likely sandbox handles from the middleware and runtime."""

    targets: list[Any] = [middleware]
    for source in (middleware, runtime):
        if source is None:
            continue
        for attr_name in _SANDBOX_ATTR_NAMES:
            target = getattr(source, attr_name, None)
            if target is not None and target not in targets:
                targets.append(target)

    runtime_context = getattr(runtime, "context", None)
    if isinstance(runtime_context, dict):
        for attr_name in _SANDBOX_ATTR_NAMES:
            target = runtime_context.get(attr_name)
            if target is not None and target not in targets:
                targets.append(target)

    return targets


async def _arecover_missing_sandbox(
    middleware: Any,
    state: Any | None,
    runtime: Any | None,
) -> bool:
    """Best-effort async sandbox recreation/reattachment for compatible backends."""

    for target in _iter_recovery_targets(middleware, runtime):
        for method_name in _RECOVERY_METHOD_NAMES:
            method = getattr(target, method_name, None)
            if not callable(method):
                continue
            try:
                await _maybe_await(_call_with_supported_args(method, state, runtime))
                logger.info(
                    "Recovered missing skills sandbox via %s.%s",
                    target.__class__.__name__,
                    method_name,
                )
                return True
            except TypeError:
                # Signature did not match the current runtime; try another hook.
                continue
            except Exception as recovery_error:
                logger.warning(
                    "Failed to recover missing skills sandbox via %s.%s: %s",
                    target.__class__.__name__,
                    method_name,
                    recovery_error,
                )
    return False


def _recover_missing_sandbox(
    middleware: Any,
    state: Any | None,
    runtime: Any | None,
) -> bool:
    """Best-effort sync sandbox recreation/reattachment for compatible backends."""

    for target in _iter_recovery_targets(middleware, runtime):
        for method_name in _RECOVERY_METHOD_NAMES:
            method = getattr(target, method_name, None)
            if not callable(method) or inspect.iscoroutinefunction(method):
                continue
            try:
                value = _call_with_supported_args(method, state, runtime)
                if inspect.isawaitable(value):
                    continue
                logger.info(
                    "Recovered missing skills sandbox via %s.%s",
                    target.__class__.__name__,
                    method_name,
                )
                return True
            except TypeError:
                # Signature did not match the current runtime; try another hook.
                continue
            except Exception as recovery_error:
                logger.warning(
                    "Failed to recover missing skills sandbox via %s.%s: %s",
                    target.__class__.__name__,
                    method_name,
                    recovery_error,
                )
    return False


def _fallback_result() -> None:
    """Fail open so the agent can answer without dynamically listed skills."""

    return None


def _wrap_before_agent(original: _F) -> _F:
    @wraps(original)
    def before_agent_with_sandbox_recovery(self: Any, state: Any, runtime: Any) -> Any:
        try:
            return original(self, state, runtime)
        except Exception as error:
            if not _is_missing_sandbox_error(error):
                raise
            logger.warning(
                "Skills sandbox was missing while listing skills; retrying once after recovery",
                exc_info=True,
            )
            _recover_missing_sandbox(self, state, runtime)
            try:
                return original(self, state, runtime)
            except Exception as retry_error:
                if not _is_missing_sandbox_error(retry_error):
                    raise
                logger.error(
                    "Skills sandbox recovery failed; continuing without skills",
                    exc_info=True,
                )
                return _fallback_result()

    return before_agent_with_sandbox_recovery  # type: ignore[return-value]


def _wrap_abefore_agent(original: _F) -> _F:
    @wraps(original)
    async def abefore_agent_with_sandbox_recovery(
        self: Any, state: Any, runtime: Any
    ) -> Any:
        try:
            return await original(self, state, runtime)
        except Exception as error:
            if not _is_missing_sandbox_error(error):
                raise
            logger.warning(
                "Skills sandbox was missing while listing skills; retrying once after recovery",
                exc_info=True,
            )
            await _arecover_missing_sandbox(self, state, runtime)
            try:
                return await original(self, state, runtime)
            except Exception as retry_error:
                if not _is_missing_sandbox_error(retry_error):
                    raise
                logger.error(
                    "Skills sandbox recovery failed; continuing without skills",
                    exc_info=True,
                )
                return _fallback_result()

    return abefore_agent_with_sandbox_recovery  # type: ignore[return-value]


def patch_skills_middleware_missing_sandbox() -> None:
    """Patch LangChain's SkillsMiddleware if it is available in this environment."""

    skills_middleware: type[Any] | None = None
    import_errors: list[Exception] = []

    for module_name in (
        "langchain.agents.middleware",
        "langchain.agents.middleware.skills",
    ):
        try:
            module = __import__(module_name, fromlist=["SkillsMiddleware"])
        except Exception as error:
            import_errors.append(error)
            continue
        skills_middleware = getattr(module, "SkillsMiddleware", None)
        if skills_middleware is not None:
            break

    if skills_middleware is None:
        logger.debug(
            "LangChain SkillsMiddleware is unavailable; skipping sandbox recovery patch: %s",
            import_errors[-1] if import_errors else "not found",
        )
        return

    if getattr(skills_middleware, _PATCHED_ATTR, False):
        return

    before_agent = getattr(skills_middleware, "before_agent", None)
    if callable(before_agent):
        setattr(skills_middleware, "before_agent", _wrap_before_agent(before_agent))

    abefore_agent = getattr(skills_middleware, "abefore_agent", None)
    if callable(abefore_agent):
        setattr(skills_middleware, "abefore_agent", _wrap_abefore_agent(abefore_agent))

    setattr(skills_middleware, _PATCHED_ATTR, True)
    logger.info("Patched LangChain SkillsMiddleware missing-sandbox handling")


__all__ = ["patch_skills_middleware_missing_sandbox"]
