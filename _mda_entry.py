from pathlib import Path

from managed_deepagents.runtime import compile_managed_agent

from _mda_connectors import connectors as _connectors
from agent import agent as _definition
from identity import identity as _identity
from src.utils.trace_root_metadata import (
    build_docs_agent_trace_metadata,
    user_id_from_config,
)

_system_prompt = Path(__file__).with_name("instructions.md").read_text()


def agent(config):
    compiled = compile_managed_agent(
        _definition,
        config,
        system_prompt=_system_prompt,
        connectors=_connectors,
        identity=_identity,
    )
    return compiled.with_config(
        {
            "metadata": build_docs_agent_trace_metadata(
                user_id=user_id_from_config(config),
            )
        }
    )
