<general_rules>
- Always search existing modules before creating new functions. For retrieval functions, check `backend/retrieval.py` and `backend/retrieval_graph/` first. For utility functions, check `backend/utils.py`.
- When creating new LangGraph nodes or chains, place them in the appropriate subdirectory within `backend/retrieval_graph/` and follow the existing state management patterns.
- Use Ruff for backend code formatting and linting. Run `make format` to format code and `make lint` to check for issues before committing.
- Use ESLint and Prettier for frontend code. Run `yarn lint` to check for issues and `yarn format` to format code in the frontend directory.
- Always update configuration classes in `backend/configuration.py` when adding new configurable parameters rather than hardcoding values.
- Follow the existing prompt template patterns in `backend/retrieval_graph/prompts.py` when creating new prompts.
- Use the established state management patterns defined in `backend/retrieval_graph/state.py` for LangGraph implementations.
- When modifying vector store operations, update both ingestion (`backend/ingest.py`) and retrieval (`backend/retrieval.py`) components consistently.
- Environment variables should be added to `.env.gcp.yaml.example` as documentation for required configuration.
- Use the existing utility functions in `backend/utils.py` for common operations like document formatting and model loading.
</general_rules>

<repository_structure>
This repository implements a dual-architecture chatbot with separate backend and frontend applications:

**Backend (`backend/`)**: Python-based LangGraph implementation
- `retrieval_graph/`: Core LangGraph chat logic with graph definitions, state management, and prompts
  - `graph.py`: Main conversational retrieval graph implementation
  - `state.py`: State management classes for LangGraph
  - `prompts.py`: Prompt templates for various chat operations
  - `researcher_graph/`: Specialized research functionality subgraph
- `ingest.py`: Document processing and vector store ingestion pipeline
- `configuration.py`: Configurable parameters for embedding models, retrieval settings, and model selection
- `retrieval.py`: Vector store retrieval operations and document search
- `utils.py`: Shared utility functions for document formatting and model loading
- `tests/evals/`: Evaluation tests with LangSmith integration

**Frontend (`frontend/`)**: Next.js TypeScript application
- `app/`: Main application directory following Next.js 13+ app router structure
  - `components/`: React components for UI elements and chat interface
  - `api/`: API route handlers for backend communication
  - `hooks/`: Custom React hooks for state management
  - `contexts/`: React context providers for global state
  - `utils/`: Frontend utility functions and constants

**Configuration & Deployment**:
- `langgraph.json`: LangGraph Cloud deployment configuration pointing to the main graph
- `pyproject.toml`: Poetry configuration for Python dependencies and project metadata
- `_scripts/`: Evaluation and index management utilities for development and CI/CD
- `.github/workflows/`: CI/CD pipelines for linting, evaluation, and deployment
</repository_structure>

<dependencies_and_installation>
**Backend Dependencies**: Managed with Poetry (Python 3.11+ required)
- Install backend dependencies: `poetry install`
- Install with development dependencies: `poetry install --with dev`
- Key dependencies: LangChain, LangGraph, Weaviate client, OpenAI/Anthropic clients

**Frontend Dependencies**: Managed with Yarn
- Install frontend dependencies: `cd frontend && yarn install`
- Key dependencies: Next.js, React, TypeScript, Tailwind CSS, Radix UI components

**Environment Setup**:
- Copy `.env.gcp.yaml.example` to `.env` and configure required API keys
- Required: OpenAI/Anthropic API keys, Weaviate URL and API key, LangSmith API key for tracing
- Database URL required for record manager (document indexing)
</dependencies_and_installation>

<testing_instructions>
**Backend Testing**: Uses pytest for evaluation tests with LangSmith integration
- Run evaluation tests: `poetry run pytest backend/tests/evals/`
- Main test file: `backend/tests/evals/test_e2e.py` evaluates end-to-end retrieval and answer quality
- Tests measure retrieval recall, answer correctness, and context correctness with specific score thresholds
- Tests require environment variables for API keys (OpenAI, Anthropic, Weaviate, LangSmith)
- Evaluation uses LangSmith datasets and creates experiment runs for tracking model performance
- CI/CD runs tests automatically on push/PR via GitHub Actions with required secrets configured
