"""Simple in-memory cache for documentation search."""

import fnmatch
import logging
import sys
import threading
import time

logger = logging.getLogger(__name__)


class RedisCache:
    """In-memory cache with TTL support."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_entries: int = 0,
        **kwargs,  # Accept but ignore redis-specific args
    ):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._memory: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        logger.info("Using in-memory cache")

    @property
    def is_redis(self) -> bool:
        return False

    @property
    def cache_type(self) -> str:
        return "in-memory"

    def get(self, key: str) -> str | None:
        """Get value from cache."""
        with self._lock:
            if key in self._memory:
                value, ts = self._memory[key]
                if time.time() - ts < self.ttl_seconds:
                    return value
                del self._memory[key]
            return None

    def set(self, key: str, value: str, ttl_seconds: int | None = None) -> bool:
        """Set value with TTL."""
        with self._lock:
            if self.max_entries > 0 and len(self._memory) >= self.max_entries:
                oldest = min(self._memory, key=lambda k: self._memory[k][1])
                del self._memory[oldest]
            self._memory[key] = (value, time.time())
            return True

    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                return True
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists and not expired."""
        with self._lock:
            if key in self._memory:
                _, ts = self._memory[key]
                if time.time() - ts < self.ttl_seconds:
                    return True
                del self._memory[key]
            return False

    def ttl(self, key: str) -> int:
        """Get remaining TTL in seconds. Returns -1 if key doesn't exist."""
        with self._lock:
            if key in self._memory:
                _, ts = self._memory[key]
                remaining = self.ttl_seconds - (time.time() - ts)
                if remaining > 0:
                    return int(remaining)
                del self._memory[key]
            return -1

    def incr(self, key: str, amount: int = 1) -> int:
        """Increment counter atomically."""
        with self._lock:
            current = int(self._memory[key][0]) if key in self._memory else 0
            new_val = current + amount
            self._memory[key] = (str(new_val), time.time())
            return new_val

    def scan(self, cursor: int = 0, match: str | None = None, count: int = 100) -> tuple[int, list[str]]:
        """Scan keys with optional pattern matching."""
        if cursor != 0:
            return 0, []

        with self._lock:
            now = time.time()
            keys = []
            for key, (_, ts) in list(self._memory.items()):
                if now - ts >= self.ttl_seconds:
                    del self._memory[key]
                elif not match or fnmatch.fnmatch(key, match):
                    keys.append(key)
            return 0, keys

    def clear(self) -> int:
        """Clear entire cache. Returns count of entries removed."""
        with self._lock:
            count = len(self._memory)
            self._memory.clear()
            return count

    def size(self) -> int:
        """Get number of cache entries."""
        with self._lock:
            now = time.time()
            for key, (_, ts) in list(self._memory.items()):
                if now - ts >= self.ttl_seconds:
                    del self._memory[key]
            return len(self._memory)

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            size_bytes = sys.getsizeof(self._memory)
            for k, (v, ts) in self._memory.items():
                size_bytes += sys.getsizeof(k) + sys.getsizeof(v) + sys.getsizeof(ts)
            return {
                "cache_type": "in-memory",
                "total_entries": len(self._memory),
                "max_entries": self.max_entries or "unlimited",
                "cache_size_mb": round(size_bytes / 1048576, 2),
                "ttl_seconds": self.ttl_seconds,
            }
