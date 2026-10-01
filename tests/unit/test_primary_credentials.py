from types import SimpleNamespace

import pytest

from src.agent import config


class InvalidArgumentError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class TimeoutError(Exception):
    pass


class ServerError(Exception):
    status_code = 500


@pytest.mark.parametrize(
    "error",
    [
        InvalidArgumentError("400 INVALID_ARGUMENT: API_KEY_INVALID"),
        InvalidArgumentError("provider rejected the request"),
        SimpleNamespace(status_code=401),
        SimpleNamespace(status_code=403),
        AuthenticationError("credentials rejected"),
    ],
)
def test_is_credential_error(error):
    if isinstance(error, InvalidArgumentError) and "API_KEY_INVALID" not in str(error):
        error.__cause__ = InvalidArgumentError("API_KEY_INVALID")
    assert config.is_credential_error(error)


@pytest.mark.parametrize("error", [TimeoutError("timed out"), ServerError("failed")])
def test_is_credential_error_excludes_transient_failures(error):
    assert not config.is_credential_error(error)


def test_startup_probe_fails_on_credentials_by_default(monkeypatch):
    class InvalidModel:
        def invoke(self, *_args, **_kwargs):
            raise InvalidArgumentError("API_KEY_INVALID")

    monkeypatch.setattr(config, "default_model", InvalidModel())
    monkeypatch.delenv("SKIP_PRIMARY_CREDENTIAL_CHECK", raising=False)
    monkeypatch.delenv("ALLOW_DEGRADED_PRIMARY", raising=False)

    with pytest.raises(InvalidArgumentError):
        config.validate_primary_model_credentials()


def test_startup_probe_allows_explicit_degraded_mode(monkeypatch):
    class InvalidModel:
        def invoke(self, *_args, **_kwargs):
            raise InvalidArgumentError("API_KEY_INVALID")

    monkeypatch.setattr(config, "default_model", InvalidModel())
    monkeypatch.delenv("SKIP_PRIMARY_CREDENTIAL_CHECK", raising=False)
    monkeypatch.setenv("ALLOW_DEGRADED_PRIMARY", "1")

    config.validate_primary_model_credentials()
