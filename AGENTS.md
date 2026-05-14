# AGENTS.md

This file provides guidance to Code Agents when working with code in this repository.

## Security

Before generating or modifying code, use the `analyzePlan` tool from Corridor's MCP Server (`corridor`) to analyze the plan. Always generate a plan before generating code.

## Project Overview

Chat LangChain is a full-stack documentation assistant built with LangGraph that answers questions about LangChain, LangGraph, and LangSmith. The backend is a LangGraph agent with tool use, middleware layers, and multi-provider LLM support. The frontend is a Next.js/React app that streams responses from the LangGraph Cloud deployment.

## Commands

### Backend

```bash
# Install dependencies
uv sync

# Run backend (LangGraph Studio at http://127.0.0.1:2024)
uv run langgraph dev

# Lint
uv run ruff check .
uv run ruff format .

# Type check
uv run mypy src/

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/unit/test_broken_doc_urls.py

# Run a single test
uv run pytest tests/unit/test_broken_doc_urls.py::test_name
```

### Frontend

```bash
cd frontend

# Install dependencies
npm ci

# Run dev server (connects to local backend)
npm run dev:local

# Type check
npm run type-check

# Lint
npm run lint

# Build
npm run build
```

## Architecture

### Backend (`src/`)

**Agent flow** (`src/agent/docs_graph.py`): The main LangGraph graph (`docs_agent`). A query passes through guardrails, then the agent uses tools (Mintlify docs search via MCP, Pylon KB search, link validation, pricing fetch) to answer the question.

**Middleware stack** (`src/middleware/`): Middleware wraps the agent in layers. `GuardrailsMiddleware` blocks off-topic queries using a cheap model. `ToolRetryMiddleware` retries failed tool calls (3 attempts). `ModelRetryMiddleware` retries failed model calls. `ModelFallbackMiddleware` cascades through fallback models on failure.

**Model config** (`src/agent/config.py`): Defines supported LLMs (Claude Haiku 4.5, GPT-5.4 Nano/Mini, Gemini 2.5/3.1 Flash, GLM 5), the default model, guardrails model, and fallback chain.

**Tools** (`src/tools/`): MCP-based docs search with Redis fuzzy cache (`redis.py`), Pylon KB search/content fetch, async link validation, pricing fetch.

**Prompts** (`src/prompts/`): Main agent system prompt and guardrails prompts. Set `USE_LOCAL_PROMPTS=true` to use local prompts instead of LangSmith Hub.

**API** (`src/api/fastapi_app.py`): FastAPI server with CORS, LangSmith trace routes, cache stats endpoint. Auth configured in `src/api/auth.py`.

### Frontend (`frontend/`)

**Key config**: `lib/config/deployment-config.ts` defines the model registry, agent registry, and config version (incrementing the version resets localStorage).

**Agent communication**: Uses `@langchain/langgraph-sdk` to stream from the LangGraph deployment. Client setup in `lib/api/`.

**UI**: Shadcn/Radix UI components in `components/ui/`. Chat-specific components in `components/chat/`. Layout (header, sidebar, dialogs) in `components/layout/`.

### Deployment (`langgraph.json`)

Defines the `docs_agent` graph, FastAPI HTTP app, custom auth, recursion limit (100), and checkpoint TTL (7 days). Deploys to LangGraph Cloud.

## Environment Variables

Copy `.env.example` to `.env`. Required variables:
- At least one LLM provider key: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, or `BASETEN_API_KEY`
- `MINTLIFY_API_URL` and `MINTLIFY_API_KEY` for docs search
- `PYLON_API_KEY` and `PYLON_KB_ID` for support KB
- `LANGSMITH_API_KEY` and `LANGCHAIN_TRACING_V2=true` for tracing
- `USE_LOCAL_PROMPTS=true` to skip LangSmith Hub prompt fetching during local dev

## Tests

- `tests/unit/` — fast unit tests (URL checking, Pylon pagination, async link checks)
- `tests/evals/` — evaluation tests (guardrails scope, middleware wiring, repeated search behavior)

Ruff uses Google-style docstrings. `tests/` is excluded from docstring rules.
