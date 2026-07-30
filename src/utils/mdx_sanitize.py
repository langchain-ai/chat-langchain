"""Strip Mintlify/MDX authoring artifacts out of raw docs page source."""

from __future__ import annotations

import re

#: Public docs host used to absolutize root-relative links found in `.mdx` source.
DOCS_BASE_URL = "https://docs.langchain.com"

#: Root-relative link prefixes that only resolve on docs.langchain.com.
RELATIVE_LINK_ROOTS = ("oss", "langsmith", "labs")

#: Mintlify components that wrap prose the agent must rewrite in its own words.
COMPONENT_TAGS = (
    "Tip",
    "Note",
    "Warning",
    "Info",
    "CodeGroup",
    "Tabs",
    "Tab",
    "Accordion",
    "AccordionGroup",
    "Card",
    "CardGroup",
)

_CODE_ANNOTATION_RE = re.compile(r"[ \t]*(?:#|//)?[ \t]*\[!code[^\]]*\]")

_COMPONENT_TAG_RE = re.compile(
    # Longest name first so `<CardGroup>` is not matched as `<Card` + junk.
    r"</?(?:" + "|".join(sorted(COMPONENT_TAGS, key=len, reverse=True)) + r")(?:\s[^>]*?)?/?>",
)

_RELATIVE_LINK_RE = re.compile(
    r"\]\((/(?:" + "|".join(RELATIVE_LINK_ROOTS) + r")/[^)\s]*)\)",
)

_FENCE_THEME_MARKER = "theme={"


def _strip_fence_theme(line: str) -> str:
    """Drop a ``theme={...}`` metadata blob from a code-fence header line."""
    start = line.find(_FENCE_THEME_MARKER)
    if start == -1:
        return line

    depth = 0
    for index in range(start + len(_FENCE_THEME_MARKER) - 1, len(line)):
        char = line[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return (line[:start] + line[index + 1 :]).rstrip()
    return line[:start].rstrip()


def strip_mdx_artifacts(text: str) -> str:
    """Remove docs-build artifacts from raw `.mdx` source so it is safe to quote."""
    if not text:
        return text

    cleaned_lines: list[str] = []
    for line in text.split("\n"):
        cleaned = _CODE_ANNOTATION_RE.sub("", line)
        if cleaned.lstrip().startswith("```"):
            cleaned = _strip_fence_theme(cleaned)

        unwrapped = _COMPONENT_TAG_RE.sub("", cleaned)
        if unwrapped != cleaned and not unwrapped.strip():
            # The line held nothing but component tags; keep the prose spacing tidy.
            continue

        cleaned_lines.append(
            _RELATIVE_LINK_RE.sub(rf"]({DOCS_BASE_URL}\1)", unwrapped)
        )

    return "\n".join(cleaned_lines)


__all__ = ["COMPONENT_TAGS", "DOCS_BASE_URL", "RELATIVE_LINK_ROOTS", "strip_mdx_artifacts"]
