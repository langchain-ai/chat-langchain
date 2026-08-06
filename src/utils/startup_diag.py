"""Cold-start diagnostics. Opt-in via LG_STARTUP_DIAG=1; no-op otherwise.

Answers three questions that wall-clock timing alone cannot:

1. Is the environment slow, or is the app slow? A syscall microbenchmark plus a
   pure-CPU loop. Under gVisor, syscalls are intercepted in userspace, so a high
   syscall:CPU ratio points at the sandbox rather than at our code.
2. Is bytecode precompiled in this image? Counts .py vs __pycache__/.pyc in
   site-packages. Importing N modules without .pyc pays a compile per module.
3. Which modules are expensive, and why? Per-module self time alongside counts
   of file opens and socket calls, so each hot module can be classified as
   filesystem-bound, CPU-bound, or network-bound.

Everything is wrapped so diagnostics can never break startup.
"""

from __future__ import annotations

import os
import sys
import time

_ENABLED = os.environ.get("LG_STARTUP_DIAG", "").strip().lower() in ("1", "true", "yes")
_TAG = "[startup-diag]"

_records: list[dict] = []
_stack: list[dict] = []
_opens: dict[str, int] = {}
_sockets: dict[str, int] = {}
_reported = False
_t0 = time.perf_counter()


def _emit(line: str) -> None:
    # print, not logging: this runs before app logging is configured.
    sys.stdout.write(f"{_TAG} {line}\n")
    sys.stdout.flush()


# ----------------------------------------------------------------- env probe


def _probe_env() -> None:
    """Microbenchmark syscalls vs pure CPU. The ratio is the sandbox signature."""
    n = 2000
    target = __file__

    t = time.perf_counter()
    for _ in range(n):
        os.stat(target)
    stat_us = (time.perf_counter() - t) / n * 1e6

    t = time.perf_counter()
    for _ in range(200):
        fd = os.open(target, os.O_RDONLY)
        os.read(fd, 4096)
        os.close(fd)
    open_us = (time.perf_counter() - t) / 200 * 1e6

    t = time.perf_counter()
    x = 0
    for i in range(2_000_000):
        x += i * i
    cpu_ms = (time.perf_counter() - t) * 1000

    _emit(
        f"env: stat={stat_us:.1f}us/call open+read+close={open_us:.1f}us/call "
        f"cpu_loop={cpu_ms:.0f}ms ratio_stat_per_cpu_ms={stat_us / max(cpu_ms, 0.001):.3f}"
    )
    _emit(
        f"env: python={sys.version.split()[0]} platform={sys.platform} "
        f"cpus={os.cpu_count()} dont_write_bytecode={sys.dont_write_bytecode}"
    )


def _probe_bytecode() -> None:
    """Census of .py vs .pyc in site-packages: is this image precompiled?"""
    import sysconfig

    purelib = sysconfig.get_paths().get("purelib")
    if not purelib or not os.path.isdir(purelib):
        _emit("bytecode: site-packages not found, skipping census")
        return

    py = pyc = legacy = 0
    for root, dirs, files in os.walk(purelib):
        if os.path.basename(root) == "__pycache__":
            pyc += sum(1 for f in files if f.endswith(".pyc"))
            continue
        dirs[:] = [d for d in dirs if d != ".git"]
        for f in files:
            if f.endswith(".py"):
                py += 1
            elif f.endswith(".pyc"):
                legacy += 1

    pct = (pyc / py * 100) if py else 0.0
    _emit(
        f"bytecode: {py} .py, {pyc} __pycache__/.pyc ({pct:.0f}% coverage), "
        f"{legacy} legacy .pyc beside source -- root={purelib}"
    )
    if py and pct < 50:
        _emit(
            "bytecode: WARNING low .pyc coverage -- modules are being compiled "
            "from source at every process start"
        )


# ------------------------------------------------------------- import timing


def _current() -> str:
    return _stack[-1]["name"] if _stack else "<not-importing>"


def _audit(event: str, _args) -> None:
    try:
        if event == "open":
            mod = _current()
            _opens[mod] = _opens.get(mod, 0) + 1
        elif event in ("socket.connect", "socket.getaddrinfo"):
            mod = _current()
            _sockets[mod] = _sockets.get(mod, 0) + 1
    except Exception:
        pass


class _Loader:
    def __init__(self, inner, name):
        self._inner = inner
        self._name = name

    def create_module(self, spec):
        if hasattr(self._inner, "create_module"):
            return self._inner.create_module(spec)
        return None

    def exec_module(self, module):
        rec = {"name": self._name, "depth": len(_stack), "t": time.perf_counter()}
        _stack.append(rec)
        _records.append(rec)
        try:
            if hasattr(self._inner, "exec_module"):
                self._inner.exec_module(module)
            else:
                self._inner.load_module(self._name)
        finally:
            rec["cum"] = time.perf_counter() - rec["t"]
            while _stack and _stack[-1] is not rec:
                _stack.pop()
            if _stack:
                _stack.pop()
            if _stack:
                p = _stack[-1]
                p["child"] = p.get("child", 0.0) + rec["cum"]
            rec["self"] = rec["cum"] - rec.get("child", 0.0)

    def __getattr__(self, item):
        # Delegate get_data/get_filename/etc. so wrapping stays transparent.
        return getattr(self._inner, item)


class _Finder:
    def __init__(self, inner):
        self._inner = inner

    def find_spec(self, fullname, path=None, target=None):
        fs = getattr(self._inner, "find_spec", None)
        if fs is None:
            return None
        spec = fs(fullname, path, target)
        if spec is None or spec.loader is None or isinstance(spec.loader, _Loader):
            return spec
        spec.loader = _Loader(spec.loader, fullname)
        return spec

    def __getattr__(self, item):
        # Must forward find_distributions or importlib.metadata goes blind.
        return getattr(self._inner, item)


def start() -> None:
    """Install diagnostics. Call as early as possible in the entrypoint."""
    if not _ENABLED:
        return
    try:
        _emit("enabled (LG_STARTUP_DIAG=1)")
        _probe_env()
        _probe_bytecode()
        sys.meta_path[:] = [
            f if isinstance(f, _Finder) else _Finder(f) for f in sys.meta_path
        ]
        sys.addaudithook(_audit)
        _watch_for_quiet()
        _emit("import timer installed")
    except Exception as exc:  # never break startup
        _emit(f"failed to install: {exc!r}")


def _watch_for_quiet() -> None:
    """Report once imports go quiet, so no call at the end of the entrypoint is needed."""
    import threading

    quiet_secs = float(os.environ.get("LG_STARTUP_DIAG_QUIET_SECS", "8"))
    state = {"n": -1, "since": time.perf_counter()}

    def tick() -> None:
        if _reported:
            return
        n = len(_records)
        now = time.perf_counter()
        if n != state["n"]:
            state["n"] = n
            state["since"] = now
        elif now - state["since"] >= quiet_secs:
            report()
            return
        t = threading.Timer(2.0, tick)
        t.daemon = True
        t.start()

    t = threading.Timer(2.0, tick)
    t.daemon = True
    t.start()


def report(top_n: int = 30) -> None:
    """Log the ranked import report. Idempotent."""
    global _reported
    if not _ENABLED or _reported:
        return
    _reported = True
    try:
        done = [r for r in _records if "self" in r]
        wall = time.perf_counter() - _t0
        tot_opens = sum(_opens.values())
        tot_sockets = sum(_sockets.values())
        _emit(
            f"summary: {len(done)} modules in {wall:.2f}s since diag start, "
            f"{tot_opens} file opens, {tot_sockets} socket calls"
        )
        if tot_sockets:
            _emit("network during import (module -> socket calls):")
            for mod, n in sorted(_sockets.items(), key=lambda kv: -kv[1])[:15]:
                _emit(f"  {n:>5} sock  {mod}")
        else:
            _emit("network during import: NONE")

        _emit(f"top {top_n} modules by self time (self_ms, cum_ms, opens, socks):")
        for r in sorted(done, key=lambda r: -r["self"])[:top_n]:
            nm = r["name"]
            _emit(
                f"  {r['self'] * 1000:>8.1f}ms self {r['cum'] * 1000:>8.1f}ms cum "
                f"{_opens.get(nm, 0):>4} op {_sockets.get(nm, 0):>3} sk  {nm}"
            )
    except Exception as exc:
        _emit(f"report failed: {exc!r}")


__all__ = ["report", "start"]


# Importing this module is enough to arm it, so integration is a single import
# line with no statement between imports (keeps E402/isort happy) and no call
# needed at the end of the entrypoint.
start()
