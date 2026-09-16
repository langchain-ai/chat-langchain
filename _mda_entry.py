from managed_deepagents.runtime import compile_managed_agent

from _mda_connectors import connectors as _connectors
from agent import agent as _definition
from identity import identity as _identity
from src.prompts.docs_agent_prompt import docs_agent_prompt


def agent(config):
    return compile_managed_agent(
        _definition,
        config,
        system_prompt=docs_agent_prompt,
        connectors=_connectors,
        identity=_identity,
    )
