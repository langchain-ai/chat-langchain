"""Prompt template for the docs agent."""

from pathlib import Path

docs_agent_prompt = (Path(__file__).parents[2] / "instructions.md").read_text()
