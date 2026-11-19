"""Guard against catastrophic backtracking in regexes that run on untrusted text.

Every module-level compiled pattern in the middleware and tools runs on user input,
model output or fetched pages, inside the event loop. A super-linear pattern there
freezes the loop (and the GIL), so probes stop answering and the replica is killed
(seen Nov 2026: link-validation _URL on unclosed markdown links, ingress _SECRET_RE
on repeated 'key='). This test times each pattern on adversarial strings.
"""

from __future__ import annotations

import importlib
import re
import time

import pytest

MODULES = [
    "src.middleware.link_validation_middleware",
    "src.middleware.legal_handoff_middleware",
    "src.middleware.reply_language_middleware",
    "src.middleware.ingress_guards_middleware",
    "src.middleware.search_result_trim_middleware",
    "src.tools.link_check_tools",
    "src.tools.integration_tools",
    "src.tools.pricing_tools",
]

N = 20_000
BUDGET_S = 1.0

UNITS = [
    "a", " ", "key=", "token: ", "sk-", "[x](https://a.b/", "](https://a.b/c",
    "https://a.b/(", "（", "）", "(", ")", "[", "`", "``", "<", "<title>",
    "a=", "Bearer ", "\n", "-", "_", "/", ".", "\\", "*", "#",
]
ADVERSARIAL = [u * (N // len(u)) for u in UNITS]
ADVERSARIAL += [
    "[文档](https://docs.langchain.com/" + "a" * N + "）",
    "https://" + "a(" * (N // 2),
    "password=" + "x" * N + "!",
]


def _patterns():
    found = []
    for mod_name in MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ModuleNotFoundError:
            continue
        for name, val in vars(mod).items():
            if isinstance(val, re.Pattern):
                found.append(pytest.param(val, id=f"{mod_name.rsplit('.', 1)[-1]}.{name}"))
    return found


@pytest.mark.parametrize("pattern", _patterns())
def test_pattern_is_fast_on_adversarial_input(pattern: re.Pattern) -> None:
    for text in ADVERSARIAL:
        t0 = time.perf_counter()
        for _ in pattern.finditer(text):
            pass
        pattern.sub("", text)
        elapsed = time.perf_counter() - t0
        assert elapsed < BUDGET_S, (
            f"{pattern.pattern[:80]!r} took {elapsed:.2f}s on {text[:30]!r}... (len {len(text)})"
        )
