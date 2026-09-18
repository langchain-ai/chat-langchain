"""Tests for the check_code_blocks tool.

The bug: the agent emitted a ```python block containing a JS `//` comment and a
```json `langgraph.json` whose `[` array was closed with `)`. Nothing validated
that the code inside a fence parses as the language the fence declares.
"""

from src.tools.code_check_tools import check_code_blocks


def _invoke(markdown: str) -> list[dict]:
    return check_code_blocks.invoke({"markdown": markdown})


def test_valid_python_block_is_ok():
    markdown = """**Answer**

```python
from langchain.agents import create_agent

agent = create_agent(model="openai:gpt-4o")
```
"""
    results = _invoke(markdown)

    assert len(results) == 1
    assert results[0]["language"] == "python"
    assert results[0]["ok"] is True
    assert results[0]["error"] is None


def test_python_block_with_js_comment_is_reported():
    markdown = """```python
// Create the agent
agent = create_agent(model="openai:gpt-4o")
```
"""
    results = _invoke(markdown)

    assert len(results) == 1
    assert results[0]["language"] == "python"
    assert results[0]["ok"] is False
    assert results[0]["error"]


def test_valid_json_block_is_ok():
    markdown = """```json
{
  "dependencies": ["."],
  "graphs": {"agent": "./agent.py:agent"}
}
```
"""
    results = _invoke(markdown)

    assert len(results) == 1
    assert results[0]["language"] == "json"
    assert results[0]["ok"] is True


def test_json_block_with_mismatched_bracket_is_reported():
    markdown = """```json
{
  "dependencies": ["."),
  "graphs": {"agent": "./agent.py:agent"}
}
```
"""
    results = _invoke(markdown)

    assert len(results) == 1
    assert results[0]["language"] == "json"
    assert results[0]["ok"] is False
    assert results[0]["error"]


def test_uniformly_indented_python_block_is_ok():
    """Snippets copied out of MDX <Tab> markup are indented but still valid."""
    markdown = """```python
    from langchain.agents import create_agent

    agent = create_agent(model="openai:gpt-4o")
```
"""
    results = _invoke(markdown)

    assert len(results) == 1
    assert results[0]["ok"] is True


def test_unparseable_languages_are_skipped():
    markdown = """```bash
pip install langchain && (
```

```
some plain text )
```

```typescript
const agent = createAgent({ ;
```
"""
    assert _invoke(markdown) == []


def test_block_index_tracks_all_fences():
    markdown = """```bash
pip install langchain
```

```python
// nope
```
"""
    results = _invoke(markdown)

    assert len(results) == 1
    assert results[0]["block_index"] == 1
