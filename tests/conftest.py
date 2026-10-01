import os


def pytest_configure() -> None:
    os.environ.setdefault("GOOGLE_API_KEY", "unit-test-google-key")
    os.environ.setdefault("OPENAI_API_KEY", "unit-test-openai-key")
    os.environ.setdefault("ANTHROPIC_API_KEY", "unit-test-anthropic-key")
    os.environ.setdefault("SKIP_PRIMARY_CREDENTIAL_CHECK", "1")
