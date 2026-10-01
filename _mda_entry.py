from pathlib import Path

from managed_deepagents.runtime import compile_managed_agent

from _mda_connectors import connectors as _connectors
from agent import agent as _definition
from identity import identity as _identity
from src.agent.config import preflight_default_model

preflight_default_model()

_system_prompt = Path(__file__).with_name("instructions.md").read_text()


def agent(config):
    return compile_managed_agent(
        _definition,
        config,
        system_prompt=_system_prompt,
        connectors=_connectors,
        identity=_identity,
    )
