"""Guest minting and end-user runs are rate limited (Nov 17 burst/probe finding)."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from managed_deepagents import _identity_auth as ia

import rate_limit
from rate_limit import SlidingWindowLimiter, client_ip


def _ctx(user, action="create_run"):
    return SimpleNamespace(user=user, resource="threads", action=action, permissions=[])


class TestLimiter(unittest.TestCase):
    def test_per_key_window(self):
        lim = SlidingWindowLimiter([(3, 60.0)])
        self.assertIsNone(lim.hit("a", now=0))
        self.assertIsNone(lim.hit("a", now=1))
        self.assertIsNone(lim.hit("a", now=2))
        self.assertIsNotNone(lim.hit("a", now=3))
        self.assertIsNone(lim.hit("b", now=3))
        self.assertIsNone(lim.hit("a", now=61))

    def test_two_rules(self):
        lim = SlidingWindowLimiter([(10, 60.0), (40, 600.0)])
        t = 0.0
        allowed = 0
        for _ in range(100):
            if lim.hit("u", now=t) is None:
                allowed += 1
            t += 7  # ~8.5/min, under the per-minute cap
        # 100 hits over ~700s: the 10-minute cap holds it near 40 per window
        self.assertLess(allowed, 60)

    def test_real_user_pace_never_limited(self):
        lim = SlidingWindowLimiter([(10, 60.0), (40, 600.0)])
        # 5/min bursts and 18 per 10 min were the real maxima (Nov 14-16)
        for i in range(18):
            self.assertIsNone(lim.hit("u", now=i * 30.0))

    def test_client_ip(self):
        self.assertEqual(client_ip({"x-forwarded-for": "34.71.41.140, 172.22.7.130"}, "172.22.0.142"), "34.71.41.140")
        # a caller-supplied hop on the left can't change the key
        self.assertEqual(client_ip({"x-forwarded-for": "1.1.1.1,34.71.41.140, 172.22.7.130"}, "172.22.1.3"), "34.71.41.140")
        self.assertEqual(client_ip({}, "10.0.0.9"), "10.0.0.9")
        self.assertEqual(client_ip({"x-real-ip": "5.6.7.8"}, None), "5.6.7.8")


class TestRunLimit(unittest.TestCase):
    def setUp(self):
        rate_limit.install_run_limit()
        rate_limit.RUNS = SlidingWindowLimiter([(2, 60.0)])

    def _run(self, user, action="create_run"):
        return asyncio.run(ia._scope_threads(_ctx(user, action), {"metadata": {}}))

    def test_guest_runs_capped(self):
        user = SimpleNamespace(mda_user_id="guest:abc", identity="guest:abc", permissions=[])
        with patch.object(ia, "_is_unscoped_principal", return_value=False), patch(
            "managed_deepagents._identity_auth.is_studio_user", return_value=False
        ), patch.object(ia, "_thread_owner_key", return_value="guest:abc"):
            self.assertEqual(self._run(user), {"owner": "guest:abc"})
            self._run(user)
            with self.assertRaises(Exception) as cm:
                self._run(user)
            self.assertEqual(getattr(cm.exception, "status_code", None), 429)
            # reads are never limited
            self._run(user, action="read")

    def test_studio_not_capped(self):
        user = SimpleNamespace(mda_user_id="studio", identity="studio", permissions=[])
        with patch.object(ia, "_is_unscoped_principal", return_value=True):
            for _ in range(5):
                self._run(user)


class TestMintLimit(unittest.TestCase):
    def test_mint_429_after_cap(self):
        import guest_auth

        rate_limit.GUEST_MINTS = SlidingWindowLimiter([(2, 600.0)])
        req = SimpleNamespace(headers={"x-forwarded-for": "6.6.6.6, 9.9.9.9, 172.22.0.1"}, client=SimpleNamespace(host="10.0.0.1"))
        with patch.dict("os.environ", {"GUEST_AUTH_SECRET": "s"}):
            codes = [asyncio.run(guest_auth._issue_guest(req)).status_code for _ in range(3)]
            other = SimpleNamespace(headers={"x-forwarded-for": "8.8.8.8, 172.22.0.1"}, client=None)
            self.assertEqual(asyncio.run(guest_auth._issue_guest(other)).status_code, 200)
        self.assertEqual(codes, [200, 200, 429])


class TestForwardedIp(unittest.TestCase):
    def test_signed_forward_keys_on_visitor(self):
        import time as _t

        import guest_auth
        from rate_limit import forwarded_client_ip, sign_forwarded_ip

        ts = int(_t.time())
        good = {"x-guest-client-ip": "34.71.41.140", "x-guest-client-sig": f"{ts}.{sign_forwarded_ip('s', '34.71.41.140', ts)}"}
        self.assertEqual(forwarded_client_ip(good, "s"), "34.71.41.140")
        # wrong secret, stale, tampered ip, garbage -> not trusted
        self.assertIsNone(forwarded_client_ip(good, "other"))
        self.assertIsNone(forwarded_client_ip(good, "s", now=ts + 600))
        self.assertIsNone(forwarded_client_ip({**good, "x-guest-client-ip": "1.1.1.1"}, "s"))
        self.assertIsNone(forwarded_client_ip({"x-guest-client-ip": "1.1.1.1", "x-guest-client-sig": "nope"}, "s"))
        self.assertIsNone(forwarded_client_ip({}, "s"))

        # mints proxied by the website share Vercel's IP but are capped per visitor
        rate_limit.GUEST_MINTS = SlidingWindowLimiter([(2, 600.0)])
        vercel = {"x-forwarded-for": "76.76.21.21, 172.22.0.1"}

        def req(ip):
            sig = f"{ts}.{sign_forwarded_ip('s', ip, ts)}"
            return SimpleNamespace(headers={**vercel, "x-guest-client-ip": ip, "x-guest-client-sig": sig}, client=None)

        with patch.dict("os.environ", {"GUEST_AUTH_SECRET": "s"}):
            a = [asyncio.run(guest_auth._issue_guest(req("5.5.5.5"))).status_code for _ in range(3)]
            b = asyncio.run(guest_auth._issue_guest(req("7.7.7.7"))).status_code
            # a forged, unsigned header falls back to the network hop
            forged = SimpleNamespace(headers={**vercel, "x-guest-client-ip": "9.9.9.9"}, client=None)
            c = [asyncio.run(guest_auth._issue_guest(forged)).status_code for _ in range(3)]
        self.assertEqual(a, [200, 200, 429])
        self.assertEqual(b, 200)
        self.assertEqual(c, [200, 200, 429])


if __name__ == "__main__":
    unittest.main()
