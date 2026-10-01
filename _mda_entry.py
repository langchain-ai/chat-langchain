from pathlib import Path

from managed_deepagents.runtime import compile_managed_agent

from _mda_connectors import connectors as _connectors
from agent import DOCS_AGENT_RECURSION_LIMIT
from agent import agent as _definition
from identity import identity as _identity
from src.middleware.recursion_boundary import with_recursion_boundary

_system_prompt = Path(__file__).with_name("instructions.md").read_text()


def agent(config):
    compiled_agent = compile_managed_agent(
        _definition,
        config,
        system_prompt=_system_prompt,
        connectors=_connectors,
        identity=_identity,
    )
    return with_recursion_boundary(
        compiled_agent.with_config({"recursion_limit": DOCS_AGENT_RECURSION_LIMIT})
    )
