from pathlib import Path

from managed_deepagents.runtime import compile_managed_agent

from _mda_connectors import connectors as _connectors
from agent import agent as _definition
from identity import identity as _identity
from src.tools.pylon_tools import validate_pylon_configuration

_system_prompt = Path(__file__).with_name("instructions.md").read_text()


def agent(config):
    validate_pylon_configuration()
    return compile_managed_agent(
        _definition,
        config,
        system_prompt=_system_prompt,
        connectors=_connectors,
        identity=_identity,
    )
