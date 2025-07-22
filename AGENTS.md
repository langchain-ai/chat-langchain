<general_rules>
When developing code in this repository, always follow these patterns:

- Before creating new backend functions, search the backend/ directory to see if similar functionality exists. Backend modules use 'from backend.' imports for internal modules.
- When adding new retrieval or vector store functionality, check backend/retrieval.py and backend/retrieval_graph/ first.
- For document processing or ingestion features, examine backend/ingest.py and backend/parser.py before implementing.
- Frontend components should be placed in frontend/app/components/ following the existing component structure with TypeScript and Tailwind CSS.
- Always use the provided linting and formatting tools: run 'make format' and 'make lint' for Python code, and 'yarn format' for frontend code.
- Environment variables should be configured in .env files and referenced through the langgraph.json configuration.
- When modifying the LangGraph implementation, the main graph logic is in backend/retrieval_graph/graph.py.
- For LangGraph Cloud deployment, ensure changes are compatible with the langgraph.json configuration.
- Use the utilities in backend/utils.py (format_docs, load_chat_model) rather than reimplementing common functionality.
- Configuration management follows the pattern in backend/configuration.py and backend/retrieval_graph/configuration.py.
</general_rules>

<repository_structure>
This repository implements a LangChain-powered chatbot with a Python backend using LangGraph and a Next.js frontend:

**Backend (Python/LangGraph):**
- backend/retrieval_graph/graph.py: Main LangGraph implementation for conversational retrieval
- backend/ingest.py: Document ingestion pipeline using LangChain loaders and Weaviate
- backend/retrieval.py: Weaviate vector store operations and retrieval logic
- backend/configuration.py: Base configuration management
- backend/utils.py: Shared utilities (format_docs, load_chat_model)
- backend/retrieval_graph/: LangGraph-specific modules including state management and prompts
- backend/tests/evals/: LangSmith-based evaluation tests

**Frontend (Next.js/TypeScript):**
- frontend/app/: Next.js app directory with TypeScript and Tailwind CSS
- frontend/app/components/: React components for chat interface and UI elements
- frontend/app/api/: API routes for LangGraph integration
- frontend/app/contexts/: React contexts for state management

**Scripts and Infrastructure:**
- _scripts/: Evaluation scripts and index management utilities
- terraform/: Infrastructure as code for deployment
- langgraph.json: LangGraph Cloud deployment configuration
- Makefile: Python linting and formatting commands
</repository_structure>

<dependencies_and_installation>
This repository uses different package managers for backend and frontend:

**Backend Dependencies:**
- Managed with Poetry (pyproject.toml and poetry.lock)
- Install with: `poetry install --with dev`
- Python 3.11+ required
- Key dependencies: LangChain ecosystem, LangGraph, Weaviate client, pytest

**Frontend Dependencies:**
- Managed with Yarn (package.json and yarn.lock)
- Install with: `yarn install`
- Key dependencies: Next.js, React, TypeScript, Tailwind CSS, various UI libraries

**Environment Configuration:**
- Create .env file in root directory for environment variables
- Required variables include API keys for LangChain services, Weaviate URL/API key, and database URLs
- Configuration is loaded through langgraph.json for LangGraph Cloud deployment
</dependencies_and_installation>

<testing_instructions>
Testing is primarily focused on the backend with LangSmith-based evaluations:

**Backend Testing:**
- Framework: pytest with LangSmith evaluations
- Test location: backend/tests/evals/
- Run tests: `poetry run pytest backend/tests/evals`
