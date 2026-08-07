# Agent module exports (empty to avoid circular imports)
# Empty to avoid circular import issues
# Import agent directly from graph.py instead:
# from src.agent.graph import agent

# Cold-start diagnostics. Importing this arms an import timer and logs a report;
# no-op unless LG_STARTUP_DIAG=1. Also armed from src/api/__init__.py so it
# starts at whichever of graph / auth / http app the platform loads first.
from src.utils import startup_diag  # noqa: F401
