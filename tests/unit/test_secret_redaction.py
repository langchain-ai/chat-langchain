"""Tests for deterministic credential and trace-write redaction."""

import json
from unittest.mock import Mock
from uuid import uuid4

import pytest
from langsmith import Client

from src.utils.secret_redaction import (
    REDACTED_SECRET,
    create_secret_anonymizer,
    redact_secrets,
)


@pytest.mark.parametrize(
    "secret",
    [
        "lsv2_pt_" + "a" * 32,
        "lsv2_sk_" + "b" * 32,
        "lcl_" + "c" * 32,
        "sk-proj-" + "d" * 32,
        "sk-ant-" + "e" * 32,
        "sk-" + "f" * 32,
        "AIza" + "g" * 35,
        *[f"gh{kind}_" + "h" * 36 for kind in "pousr"],
        *[f"xox{kind}-123456789012-" + "j" * 24 for kind in "baprs"],
        "tvly-" + "k" * 32,
        "sk_live_" + "m" * 24,
        "rk_live_" + "n" * 24,
        "AKIA" + "A" * 16,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0In0.synthetic_signature",
    ],
)
def test_supported_credentials(secret):
    assert redact_secrets(f"Use `{secret}` please") == (
        f"Use `{REDACTED_SECRET}` please",
        1,
    )


@pytest.mark.parametrize(
    "name", ["LANGSMITH_API_KEY", "api_key", "KEY", "TOKEN", "SECRET", "PASSWORD"]
)
@pytest.mark.parametrize("quote", ["", "'", '"'])
def test_secret_assignments(name, quote):
    assert redact_secrets(f"{name}={quote}synthetic-credential{quote}") == (
        f"{name}={quote}{REDACTED_SECRET}{quote}",
        1,
    )


def test_bearer_token_and_replacement_count():
    text = "Authorization: Bearer synthetic-token\nLANGSMITH_API_KEY=lsv2_pt_synthetic"
    assert redact_secrets(text) == (
        f"Authorization: Bearer {REDACTED_SECRET}\nLANGSMITH_API_KEY={REDACTED_SECRET}",
        2,
    )


@pytest.mark.parametrize(
    "text",
    [
        "https://docs.example.com/clickhouse-disk-and-memory-pressure",
        "https://docs.example.com/ask-proj-" + "a" * 32,
        "LANGSMITH_API_KEY=your_api_key_here",
        'TOKEN="your_token_here"',
        "PASSWORD=<your password>",
        "SECRET=<...>",
        "KEY=settings.X",
        "KEY=os.environ['LANGSMITH_API_KEY']",
        'KEY=os.environ["LANGSMITH_API_KEY"]',
        'KEY=os.getenv("LANGSMITH_API_KEY")',
        "TOKEN=${ACCESS_TOKEN}",
        "Bearer <token>",
        "Bearer your_token_here",
        "lsv2_pt_your_api_key_here",
        "sk-proj-your_api_key_here_long_placeholder",
    ],
)
def test_placeholders_and_url_slugs_are_unchanged(text):
    assert redact_secrets(text) == (text, 0)


def test_redaction_is_idempotent():
    redacted, _ = redact_secrets("KEY=synthetic-key\nBearer synthetic-token")
    assert redact_secrets(redacted) == (redacted, 0)


def test_trace_anonymizer_handles_nested_inputs_outputs():
    secret = "sk-proj-" + "a" * 32
    payload = {
        "inputs": {"messages": [{"content": [{"type": "text", "text": secret}]}]},
        "outputs": {"answer": secret, "tokens": 12},
    }
    anonymized = create_secret_anonymizer()(payload)
    assert secret not in json.dumps(anonymized)
    assert REDACTED_SECRET in json.dumps(anonymized)
    assert anonymized["outputs"]["tokens"] == 12


def test_langsmith_client_redacts_create_and_update_payloads():
    secret = "lsv2_pt_" + "a" * 32
    session = Mock()
    session.request.return_value.status_code = 200
    client = Client(
        api_url="https://langsmith.invalid",
        api_key="test",
        session=session,
        auto_batch_tracing=False,
        info={"version": "0.0.0"},
        anonymizer=create_secret_anonymizer(),
    )
    run_id = uuid4()
    client.create_run("synthetic request", {"query": secret}, "chain", id=run_id)
    client.update_run(run_id, outputs={"answer": secret})

    assert session.request.call_count == 2
    for request in session.request.call_args_list:
        payload = json.loads(request.kwargs["data"])
        serialized = json.dumps(payload)
        assert secret not in serialized
        assert REDACTED_SECRET in serialized
    client.close()
