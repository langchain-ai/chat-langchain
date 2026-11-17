"""End users can't touch the long-term store (Nov 17 probe finding)."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from managed_deepagents import _identity_auth as ia

import store_guard


def _ctx(user):
    return SimpleNamespace(user=user, resource="store", action="search")


class TestStoreGuard(unittest.TestCase):
    def setUp(self):
        store_guard.install_store_guard()

    def _check(self, user, namespace=("memories", "guest:x")):
        asyncio.run(ia._scope_store(_ctx(user), {"namespace": list(namespace)}))

    def test_guest_is_refused(self):
        user = SimpleNamespace(mda_user_id="guest:abc", identity="guest:abc", permissions=[])
        with patch.object(ia, "_is_unscoped_principal", return_value=False):
            with self.assertRaises(Exception) as cm:
                self._check(user)
        self.assertEqual(getattr(cm.exception, "status_code", None), 403)

    def test_signed_in_user_is_refused(self):
        user = SimpleNamespace(mda_user_id="5f0c-user", identity="5f0c-user", permissions=[])
        with patch.object(ia, "_is_unscoped_principal", return_value=False):
            with self.assertRaises(Exception):
                self._check(user, ("anything",))

    def test_studio_principal_keeps_access(self):
        user = SimpleNamespace(mda_user_id="studio", identity="studio", permissions=[])
        with patch.object(ia, "_is_unscoped_principal", return_value=True):
            self._check(user)

    def test_identity_links_service_keeps_access(self):
        user = SimpleNamespace(mda_user_id=ia.IDENTITY_LINKS_SERVICE_USER, identity="svc", permissions=[])
        with patch.object(ia, "_is_unscoped_principal", return_value=False):
            self._check(user, (ia.IDENTITY_LINKS_STORE_PREFIX, "x"))

    def test_install_is_idempotent(self):
        store_guard.install_store_guard()
        self.assertIs(ia._assert_store_access, store_guard._guarded_assert_store_access)


if __name__ == "__main__":
    unittest.main()
