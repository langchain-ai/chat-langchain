<general_rules>
When creating new LangGraph functions or conversation flows, always first search the backend/retrieval_graph/ directory to see if similar functionality exists. If creating new graph nodes or state management, place them in backend/retrieval_graph/graph.py or create new files within the retrieval_graph module following the existing pattern.

When creating new retrieval or document processing functions, check backend/ingest.py and backend/retrieval.py first to see if similar functionality exists. These modules handle document ingestion and vector store operations respectively.

When creating new React components, follow the frontend/app/components/ structure. Place UI components in frontend/app/components/ui/ and feature-specific components in the main components directory. Always check existing components before creating new ones.

Use the provided linting and formatting commands consistently:
- Run `make format` to format code with Ruff and fix import sorting
- Run `make lint` to check code quality with Ruff
- For frontend, use `yarn format` to format with Prettier and `yarn lint` for ESLint checks

Common utility scripts are located in the _scripts/ directory:
- clear_index.py: Clears the vector store index
- evaluate_chains.py, evaluate_chains_agent.py, evaluate_chains_improved_chain.py: Various evaluation scripts
- evaluate_chat_langchain.py: Main evaluation script for the chat system

Configuration management should use the dataclass-based approach in backend/configuration.py. When adding new configuration options, extend the BaseConfiguration class and follow the existing pattern with proper type annotations and metadata.

Vector store operations should always use context managers as implemented in backend/retrieval.py to ensure proper resource cleanup.
</general_rules>

<repository_structure>
This repository implements a dual-architecture chatbot system with a Python backend and Next.js frontend.

Backend (Python/LangGraph/Poetry):
- backend/retrieval_graph/graph.py: Main LangGraph implementation defining the conversational retrieval flow
- backend/ingest.py: Document ingestion pipeline for loading and processing documentation
- backend/retrieval.py: Vector store operations and retriever implementations using context managers
- backend/configuration.py: Dataclass-based configuration management for the system
- backend/retrieval_graph/: Contains the core LangGraph modules including state management, prompts, and researcher graph

Frontend (Next.js/TypeScript/Yarn):
- frontend/app/components/: React components organized by feature and UI components in ui/ subdirectory
- frontend/app/contexts/: React contexts for state management across the application
- frontend/app/hooks/: Custom React hooks for reusable logic
- frontend/app/api/: Next.js API routes for backend communication

Deployment and Configuration:
- langgraph.json: LangGraph Cloud deployment configuration pointing to backend/retrieval_graph/graph.py:graph
- pyproject.toml: Python dependencies and Poetry configuration
- frontend/package.json: Node.js dependencies and Yarn scripts
- .github/workflows/: CI/CD pipelines for linting, evaluation, and deployment
</repository_structure>

<dependencies_and_installation>
Backend dependencies are managed with Poetry (Python 3.11+):
- Install backend dependencies: `poetry install --with dev`
- Key technologies: LangChain ecosystem (langchain, langchain-core, langchain-community, langchain-openai, langchain-anthropic, langchain-weaviate), LangGraph for conversation flow, Weaviate vector store client

Frontend dependencies are managed with Yarn:
- Install frontend dependencies: `cd frontend && yarn install`
- Key technologies: Next.js 14.2.25, React 18.2.0, TypeScript, Tailwind CSS, various UI libraries (@radix-ui, @assistant-ui)

The system uses Weaviate as the vector store for document storage and retrieval. Ensure proper environment variables are set for Weaviate connection (WEAVIATE_URL, WEAVIATE_API_KEY).

Development tools include Ruff for Python linting/formatting, Prettier for frontend formatting, and ESLint for frontend linting.
</dependencies_and_installation>

<testing_instructions>
Backend testing uses pytest framework with LangSmith evaluations:
- Run tests: `poetry run pytest backend/tests/evals`
- Main test file: backend/tests/evals/test_e2e.py contains end-to-end evaluation tests
- Test types include retrieval recall evaluation, answer correctness scoring, and context-based answer evaluation
- Tests require environment variables: LANGSMITH_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, WEAVIATE_URL, WEAVIATE_API_KEY

CI/CD testing is automated through GitHub Actions:
