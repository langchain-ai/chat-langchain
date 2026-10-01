"""Tests for authentication-aware model fallback behavior."""

import asyncio

from langchain_core.runnables import RunnableLambda

from src.middleware.model_fallback import (
    AuthenticationAwareRetry,
    AuthenticationAwareRunnableWithFallbacks,
    ModelAvailabilityState,
)


class AuthFailure(Exception):
    pass


def test_auth_failure_skips_primary_during_cooldown():
    calls = []
    state = ModelAvailabilityState(cooldown_seconds=60)

    def primary(_: str) -> str:
        calls.append("primary")
        raise AuthFailure("API_KEY_INVALID")

    primary = AuthenticationAwareRetry(
        bound=RunnableLambda(primary),
        max_attempt_number=1,
        availability=state,
        model_id="primary",
    )
    fallback = RunnableLambda(lambda _: calls.append("fallback") or "ok")
    runnable = AuthenticationAwareRunnableWithFallbacks(primary, [fallback], state)

    assert runnable.invoke("input") == "ok"
    assert runnable.invoke("input") == "ok"
    assert calls == ["primary", "fallback", "fallback"]


def test_non_auth_failure_retries_primary_on_next_call():
    calls = []

    def primary(_: str) -> str:
        calls.append("primary")
        raise RuntimeError("temporary failure")

    state = ModelAvailabilityState(cooldown_seconds=60)
    runnable = AuthenticationAwareRunnableWithFallbacks(
        AuthenticationAwareRetry(
            bound=RunnableLambda(primary),
            max_attempt_number=1,
            availability=state,
            model_id="primary",
        ),
        [RunnableLambda(lambda _: calls.append("fallback") or "ok")],
        state,
    )

    assert runnable.invoke("input") == "ok"
    assert runnable.invoke("input") == "ok"
    assert calls == ["primary", "fallback", "primary", "fallback"]


def test_auth_failure_skips_primary_async_during_cooldown():
    calls = []
    state = ModelAvailabilityState(cooldown_seconds=60)

    async def primary(_: str) -> str:
        calls.append("primary")
        raise AuthFailure("HTTP 403")

    async def fallback(_: str) -> str:
        calls.append("fallback")
        return "ok"

    runnable = AuthenticationAwareRunnableWithFallbacks(
        AuthenticationAwareRetry(
            bound=RunnableLambda(primary),
            max_attempt_number=1,
            availability=state,
            model_id="primary",
        ),
        [RunnableLambda(fallback)],
        state,
    )

    async def run() -> None:
        assert await runnable.ainvoke("input") == "ok"
        assert await runnable.ainvoke("input") == "ok"

    asyncio.run(run())
    assert calls == ["primary", "fallback", "fallback"]
