"""Close the deployment's long-term store to end users.

The managed identity auth only fences off two reserved store tables; every other
namespace is readable, writable and deletable by any authenticated caller,
guests included (found during the Nov 17, 2026 probe). Chat LangChain itself
never uses the store, so end users (guests and signed-in website users) get 403
on every store action. Studio / local-dev principals and the managed
identity-links service keep their existing access.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from managed_deepagents import _identity_auth as _ia

_INSTALLED = False


def _guarded_assert_store_access(user: Any, value: Mapping[str, object]) -> None:
    _ORIGINAL(user, value)  # keeps the managed reserved-table check
    if _ia._is_unscoped_principal(user):
        return
    if _ia._field(user, "mda_user_id") == _ia.IDENTITY_LINKS_SERVICE_USER:
        return
    raise _ia.ManagedAuthError(403, "store access is not available on this deployment")


_ORIGINAL = _ia._assert_store_access


def install_store_guard() -> None:
    """Swap in the stricter store check once (``_scope_store`` looks it up at call time)."""
    global _INSTALLED
    if _INSTALLED:
        return
    _ia._assert_store_access = _guarded_assert_store_access
    _INSTALLED = True
