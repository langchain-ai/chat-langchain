"""Tests verifying that ALLOWED_SAMPLE_RATE and the dataset description are consistent."""

import inspect
import os
import sys
from unittest.mock import MagicMock, patch

# Ensure the repo root is on the path so `src` can be imported as a package.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also create a temporary src __init__ shim if needed so Python treats it as a package.
_SRC_INIT = os.path.join(_REPO_ROOT, "src", "__init__.py")
_created_src_init = False
if not os.path.exists(_SRC_INIT):
    open(_SRC_INIT, "w").close()
    _created_src_init = True

_MIDDLEWARE_INIT = os.path.join(_REPO_ROOT, "src", "middleware", "__init__.py")
_created_middleware_init = False
if not os.path.exists(_MIDDLEWARE_INIT):
    open(_MIDDLEWARE_INIT, "w").close()
    _created_middleware_init = True


def _import_guardrails_module():
    """Import guardrails_middleware with AsyncClient patched out."""
    # Remove cached module so re-import works in isolation
    for key in list(sys.modules.keys()):
        if "guardrails_middleware" in key or "src.middleware" in key:
            del sys.modules[key]
    with patch("langsmith.AsyncClient", MagicMock()):
        import src.middleware.guardrails_middleware as mod
    return mod


def test_allowed_sample_rate_is_one_percent():
    """ALLOWED_SAMPLE_RATE must be 0.01 (1%)."""
    mod = _import_guardrails_module()
    assert mod.ALLOWED_SAMPLE_RATE == 0.01, (
        f"Expected ALLOWED_SAMPLE_RATE == 0.01, got {mod.ALLOWED_SAMPLE_RATE}"
    )


def test_add_to_dataset_description_says_one_percent():
    """The create_dataset() description in _add_to_dataset must say '1%', not '10%'."""
    mod = _import_guardrails_module()
    source = inspect.getsource(mod.GuardrailsMiddleware._add_to_dataset)

    assert "1%" in source, "Dataset description should contain '1%'"
    assert "10%" not in source, (
        "Dataset description should NOT contain '10%' — update the string to match "
        "the actual ALLOWED_SAMPLE_RATE of 0.01 (1%)"
    )


def test_inline_comment_says_one_percent():
    """The inline comment near ALLOWED_SAMPLE_RATE must reference 1%."""
    mod = _import_guardrails_module()
    source = inspect.getsource(mod)
    for line in source.splitlines():
        if "ALLOWED_SAMPLE_RATE" in line and "=" in line and "def" not in line:
            assert "1%" in line, (
                f"Inline comment on ALLOWED_SAMPLE_RATE line should mention '1%': {line!r}"
            )
            break
