"""Tests for ingress secret redaction."""

from __future__ import annotations

import pytest

from src.middleware.redaction import redact_secrets


def test_redacts_uri_password_and_preserves_surrounding_context():
    text = "connect postgres://db_user:synthPass123!@db.internal:5432/app"

    assert (
        redact_secrets(text)
        == "connect postgres://db_user:<REDACTED>@db.internal:5432/app"
    )


@pytest.mark.parametrize(
    "placeholder",
    [
        "YOUR_API_KEY_HERE",
        "<YOUR_API_KEY>",
        "${API_KEY}",
        "os.getenv('API_KEY')",
        "changeme",
        "example",
        "xxxxxxxx",
        "short",
    ],
)
def test_does_not_redact_obvious_placeholders(placeholder: str):
    text = f"password={placeholder}"

    assert redact_secrets(text) == text


def test_redacts_key_prefix_and_assignment_values():
    text = "api_key=sk-synthetic-secret-123 Bearer synthetic-bearer-token-123"

    assert redact_secrets(text) == "api_key=<REDACTED> Bearer <REDACTED>"
