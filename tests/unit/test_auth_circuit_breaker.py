import asyncio

import pytest
from langchain_core.runnables import RunnableLambda

from src.middleware.retry_middleware import (
    AuthenticationAwareRunnable,
    PrimaryModelAuthCircuit,
    is_authentication_error,
)


class GoogleApiKeyInvalidError(Exception):
    def __init__(self):
        super().__init__(
            "400 INVALID_ARGUMENT: details=[{'reason': 'API_KEY_INVALID'}]"
        )


def test_google_api_key_invalid_is_authentication_error():
    assert is_authentication_error(GoogleApiKeyInvalidError())


def test_auth_error_opens_circuit_and_skips_next_call():
    calls = 0

    def invoke(_: object) -> object:
        nonlocal calls
        calls += 1
        raise GoogleApiKeyInvalidError()

    circuit = PrimaryModelAuthCircuit("google", "gemini", cooldown_seconds=300)
    runnable = AuthenticationAwareRunnable(RunnableLambda(invoke), circuit)

    with pytest.raises(GoogleApiKeyInvalidError):
        runnable.invoke("input")
    with pytest.raises(Exception, match="gemini"):
        runnable.invoke("input")

    assert calls == 1


def test_non_auth_error_does_not_open_circuit():
    circuit = PrimaryModelAuthCircuit("google", "gemini", cooldown_seconds=300)
    runnable = AuthenticationAwareRunnable(
        RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("transient"))),
        circuit,
    )

    with pytest.raises(RuntimeError, match="transient"):
        runnable.invoke("input")

    assert circuit.allow_request()


def test_primary_is_retried_after_cooldown_expires(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("src.middleware.retry_middleware.time.monotonic", lambda: clock[0])
    circuit = PrimaryModelAuthCircuit("google", "gemini", cooldown_seconds=10)
    circuit.trip(GoogleApiKeyInvalidError())

    assert not circuit.allow_request()
    clock[0] = 110.0
    assert circuit.allow_request()


def test_async_auth_error_opens_circuit():
    calls = 0

    async def invoke(_: object) -> object:
        nonlocal calls
        calls += 1
        raise GoogleApiKeyInvalidError()

    circuit = PrimaryModelAuthCircuit("google", "gemini", cooldown_seconds=300)
    runnable = AuthenticationAwareRunnable(RunnableLambda(invoke), circuit)

    async def run() -> None:
        with pytest.raises(GoogleApiKeyInvalidError):
            await runnable.ainvoke("input")
        with pytest.raises(Exception, match="gemini"):
            await runnable.ainvoke("input")

    asyncio.run(run())
    assert calls == 1
