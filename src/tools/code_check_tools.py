"""Code block validation tool for checking fenced snippets parse before responding."""

import ast
import json
import re
import textwrap

from langchain.tools import tool

FENCE_PATTERN = re.compile(r"```([a-zA-Z]*)\n(.*?)```", re.S)

PYTHON_LANGUAGES = {"python", "py"}
JSON_LANGUAGES = {"json"}


def _check_python(source: str) -> str | None:
    """Return a syntax error description for a Python snippet, or None if it parses."""
    try:
        ast.parse(source)
    except SyntaxError as e:
        return f"{e.msg} (line {e.lineno})"
    return None


def _check_json(source: str) -> str | None:
    """Return a syntax error description for a JSON snippet, or None if it parses."""
    try:
        json.loads(source)
    except json.JSONDecodeError as e:
        return str(e)
    return None


@tool
def check_code_blocks(markdown: str) -> list[dict]:
    """Check that every fenced code block in a drafted response parses as its declared language.

    Args:
        markdown: The drafted response text containing fenced code blocks.

    Returns:
        One record per checkable block with its language, index, ok flag, and error.
        Blocks whose fence language cannot be parsed here (bash, text, shell,
        typescript, or no language) are skipped.
    """
    results: list[dict] = []

    for index, (language, body) in enumerate(FENCE_PATTERN.findall(markdown)):
        lang = language.lower()
        # Snippets copied out of MDX <Tab> markup are uniformly indented but
        # otherwise valid, so dedent before parsing to avoid false positives.
        source = textwrap.dedent(body)

        if lang in PYTHON_LANGUAGES:
            error = _check_python(source)
        elif lang in JSON_LANGUAGES:
            error = _check_json(source)
        else:
            continue

        results.append(
            {
                "language": lang,
                "block_index": index,
                "ok": error is None,
                "error": error,
            }
        )

    return results
