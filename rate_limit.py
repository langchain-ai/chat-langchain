"""Per-replica rate limits for public callers.

Nothing capped guest traffic before (Nov 17, 2026): one guest pass fired 60
runs in four minutes, and a probe rotated 151 guest ids, each minted for free
at ``POST /identity/guest``. Two cheap fences, kept in memory per replica:

* guest pass minting, per client IP (loose: the docs embed mints through
  its own server-side proxy at ui-patterns.langchain.com, so every new docs
  reader shares a few Vercel egress IPs. At 20 per 10 min that cap turned
  away ~54 real readers in 2 hours on Nov 17; real mint volume is ~100 per
  10 min across all of them);
* new runs, per end-user id (guests and signed-in website users).

Real traffic Nov 14-16 peaked at 5 runs/min and 18 runs/10 min for one id, so
the run limits sit well above any real user. Studio, local dev and managed
service principals are never limited.
"""

from __future__ import annotations

import ipaddress
import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Mapping
from typing import Any

from managed_deepagents import _identity_auth as _ia


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class SlidingWindowLimiter:
    """Allow at most ``limit`` hits per ``window`` seconds for each key."""

    def __init__(self, rules: list[tuple[int, float]], max_keys: int = 50_000) -> None:
        self.rules = rules
        self.horizon = max(window for _, window in rules)
        self.max_keys = max_keys
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, now: float | None = None) -> float | None:
        """Record a hit; return ``None`` when allowed, else seconds to wait."""
        current = time.monotonic() if now is None else now
        with self._lock:
            if len(self._hits) > self.max_keys:
                self._prune(current)
            hits = self._hits[key]
            while hits and hits[0] <= current - self.horizon:
                hits.popleft()
            for limit, window in self.rules:
                recent = [t for t in hits if t > current - window]
                if len(recent) >= limit:
                    return max(1.0, recent[0] + window - current)
            hits.append(current)
            return None

    def _prune(self, current: float) -> None:
        for key in list(self._hits):
            hits = self._hits[key]
            while hits and hits[0] <= current - self.horizon:
                hits.popleft()
            if not hits:
                del self._hits[key]


GUEST_MINTS = SlidingWindowLimiter([(_env_int("GUEST_MINTS_PER_10MIN_PER_IP", 300), 600.0)])
RUNS = SlidingWindowLimiter(
    [
        (_env_int("RUNS_PER_MIN_PER_USER", 10), 60.0),
        (_env_int("RUNS_PER_10MIN_PER_USER", 40), 600.0),
    ]
)


def _is_internal(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return True
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


def client_ip(headers: Mapping[str, str], fallback: str | None) -> str:
    """Rightmost public hop in ``X-Forwarded-For``.

    The platform's load balancer appends the real client address, then internal
    proxies append their own private hops (seen on a preview, Nov 17:
    ``<client>, 172.22.x.x``). Anything to the left of the client address is
    whatever the caller sent, so reading the first hop would let anyone pick
    their own key.
    """
    hops = [hop.strip() for hop in (headers.get("x-forwarded-for") or "").split(",") if hop.strip()]
    for hop in reversed(hops):
        if not _is_internal(hop):
            return hop
    return (headers.get("x-real-ip") or "").strip() or (fallback or "unknown")


def _is_limited_user(user: Any) -> bool:
    if _ia._is_unscoped_principal(user):
        return False
    if _ia._field(user, "mda_user_id") == _ia.IDENTITY_LINKS_SERVICE_USER:
        return False
    if _ia._channel_conversation_owner_key(user) is not None:
        return False
    return True


_ORIGINAL_SCOPE_THREADS = _ia._scope_threads
_INSTALLED = False


async def _limited_scope_threads(ctx: Any, value: Any) -> Any:
    result = await _ORIGINAL_SCOPE_THREADS(ctx, value)
    if ctx.action == "create_run" and _is_limited_user(ctx.user):
        key = _ia._thread_owner_key(ctx.user)
        if key:
            wait = RUNS.hit(key)
            if wait is not None:
                raise _ia._http_exception(
                    _ia.ManagedAuthError(429, f"Too many requests. Try again in {int(wait)}s.")
                )
    return result


def install_run_limit() -> None:
    """Swap in the rate-limited threads handler before ``build_managed_auth`` runs.

    The generated auth module imports ``identity`` (which calls this) and then
    calls ``build_managed_auth``, which registers ``_scope_threads`` by name.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _ia._scope_threads = _limited_scope_threads
    _INSTALLED = True
