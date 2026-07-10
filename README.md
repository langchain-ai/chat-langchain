# Chat LangChain

> A documentation assistant deployed as a Managed Deep Agent.

[![LangGraph](https://img.shields.io/badge/Built%20with-LangGraph-blue)](https://langchain-ai.github.io/langgraph/)
[![Python](https://img.shields.io/badge/Python-3.11+-green)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

## Overview

This is a documentation assistant agent that helps answer questions about LangChain, LangGraph, and LangSmith. It demonstrates how to build a production-ready agent using:

- **Managed Deep Agents** - For managed deployment, identity, and connectors
- **LangChain Agents** - For agent creation with middleware support
- **Guardrails** - To keep conversations on-topic

The repo also includes a Next.js frontend in `frontend/` for the public chat UI.

## Features

- **Documentation Search** - Searches official LangChain docs
- **Support KB** - Searches the Pylon knowledge base for known issues
- **Link Validation** - Verifies URLs before including in responses
- **Guardrails** - Filters off-topic queries

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

```bash
# Clone the repository
git clone https://github.com/langchain-ai/chat-langchain.git
cd chat-langchain

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
```

#### Required Environment Variables

| Variable            | Description                                                                                              |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| `ANTHROPIC_API_KEY` | Anthropic API key (or use another provider)                                                              |
| `MINTLIFY_API_URL`  | Mintlify API base URL for docs search (e.g. `https://api-dsc.mintlify.com/v1/search/docs.langchain.com`) |
| `MINTLIFY_API_KEY`  | Mintlify API key for docs search                                                                         |
| `PYLON_API_KEY`     | Pylon API key for support KB                                                                             |
| `PYLON_KB_ID`       | Pylon knowledge base ID for support articles                                                             |
| `USE_LOCAL_PROMPTS` | Optional. Set to `true` to use local prompt files instead of pulling Prompt Hub prompts                  |

### Running Locally

#### Backend

```bash
# Build the Managed Deep Agent bundle
uv run mda dev .

# Or with pip
mda dev .
```

#### Frontend

```bash
cd frontend
npm ci
npm run dev:local
```

Point the frontend at the local MDA deployment via `NEXT_PUBLIC_LANGGRAPH_API_URL`
(see `frontend/.env.local.example`). Auth, guest issuance, and LangSmith
operations go through the managed identity and connector surface.

## Project Structure

```txt
├── agent.py                    # Managed Deep Agent entrypoint
├── identity.py                 # MDA identity contract (Supabase + guest)
├── instructions.md             # Managed Deep Agent system prompt
├── connectors/
│   ├── langsmith.py            # LangSmith feedback + trace connector
│   └── mcp.py                  # Managed MCP connector declaration
├── src/
│   ├── agent/
│   │   ├── docs_graph.py      # Legacy LangGraph agent module retained for now
│   │   └── config.py          # Model configuration
│   ├── tools/
│   │   ├── docs_tools.py      # Documentation search
│   │   ├── pylon_tools.py     # Support KB tools
│   │   └── link_check_tools.py # URL validation
│   ├── prompts/
│   │   └── docs_agent_prompt.py
│   └── middleware/
│       ├── guardrails_middleware.py
│       ├── ingress_guards_middleware.py
│       └── retry_middleware.py
├── frontend/                  # Next.js public chat UI
└── pyproject.toml             # Python project config
```

## How It Works

The agent uses a docs-first research strategy:

1. **Guardrails Check** - Validates the query is LangChain-related
2. **Documentation Search** - Searches official docs via Mintlify
3. **Knowledge Base** - Searches Pylon for known issues/solutions
4. **Link Validation** - Verifies any URLs before including them
5. **Response Generation** - Synthesizes a helpful answer

## Deployment

### Managed Deep Agents

```bash
mda deploy .
```

What MDA owns in this deployment:

- **Identity** — `identity.py` verifies Supabase access tokens (multi-region) and
  issues/verifies guest tokens via `POST /identity/guest`.
- **HTTP surface** — managed ingress; no custom FastAPI app.
- **LangSmith browser ops** — `connectors/langsmith.py` proxies feedback and
  trace read/share so `LANGSMITH_API_KEY` never reaches the browser.
- **Thread titles** — generated in the browser (deterministic truncation); no
  custom `/generate-title` route.
- **Checkpointer** — managed by the Managed Deep Agents runtime.

## Resources

- [LangChain Documentation](https://docs.langchain.com/oss/python/langchain/overview)
- [LangGraph Documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [LangSmith Documentation](https://docs.langchain.com/langsmith/home)

## License

MIT
