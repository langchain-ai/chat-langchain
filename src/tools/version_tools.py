"""Tool for resolving the current released version of a LangChain ecosystem package."""

import logging
import threading
import time
from typing import Literal

import httpx
from langchain.tools import tool

logger = logging.getLogger(__name__)

TIMEOUT = 8.0
USER_AGENT = "LangChain-SupportAgent/1.0"

# In-process TTL cache keyed by (ecosystem, normalized package). Releases land a
# few times a week at most, so 5 minutes is plenty fresh and keeps repeat
# questions in one thread off the public registries.
_CACHE_TTL_SECONDS = 300
_cache_lock = threading.Lock()
_cache: dict[tuple[str, str], tuple[str, float]] = {}

# Without this allowlist the tool degenerates into a generic outbound HTTP fetch
# driven by model-generated input.
ALLOWED_PREFIXES = ("langchain", "langgraph", "langsmith", "deepagents", "@langchain/")


def _normalize(package: str) -> str:
    """Lowercase and strip a package name for allowlist checks and cache keys."""
    return package.strip().lower()


@tool
async def get_package_version(package: str, ecosystem: Literal["pypi", "npm"]) -> str:
    """ALWAYS use this tool for ANY question about the latest or current released version of a LangChain ecosystem package.

    DO NOT use docs search for version questions — the docs do not contain released
    version numbers. NEVER answer with an install/upgrade command instead of a version.

    Triggers: "latest version", "current version", "what version is", "which version",
    "newest release", "最新版", "最新版本", "小版本", "版本号".

    Args:
        package: Package name, e.g. "langgraph" or "@langchain/core".
        ecosystem: "pypi" for Python packages, "npm" for JavaScript packages.

    Returns the current released version, or a message saying it could not be resolved.
    """
    normalized = _normalize(package)
    if not normalized.startswith(ALLOWED_PREFIXES):
        return (
            f"Could not resolve version: '{package}' is not a LangChain ecosystem "
            "package. This tool only looks up langchain, langgraph, langsmith, "
            "deepagents, and @langchain/* packages."
        )

    key = (ecosystem, normalized)
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None and (now - cached[1]) < _CACHE_TTL_SECONDS:
            return cached[0]

    if ecosystem == "pypi":
        url = f"https://pypi.org/pypi/{package}/json"
    else:
        url = f"https://registry.npmjs.org/{package}"

    # Every failure path must return a string: the model needs to be able to tell
    # the user "I can't look this up" rather than see a tool error.
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = await client.get(url, timeout=TIMEOUT)
            response.raise_for_status()
            payload = response.json()
        if ecosystem == "pypi":
            version = payload.get("info", {}).get("version")
        else:
            version = payload.get("dist-tags", {}).get("latest")
    except Exception as e:
        logger.warning(f"Version lookup failed for {package} ({ecosystem}): {e}")
        return f"Could not resolve the current version of {package} ({ecosystem})."

    if not version:
        return f"Could not resolve the current version of {package} ({ecosystem})."

    result = f"{package} current released version ({ecosystem}): {version}"
    with _cache_lock:
        _cache[key] = (result, time.monotonic())
    return result
