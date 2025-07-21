<general_rules>
When creating new backend functions, always first search the backend/ directory to see if similar functionality exists before implementing new code. Use the existing patterns and avoid duplication.

All backend imports must use the 'from backend.' pattern (e.g., 'from backend.utils import format_docs', 'from backend.configuration import BaseConfiguration'). This maintains consistency across the codebase.

Before committing any changes, always run 'make format' to format code with ruff and fix import ordering, followed by 'make lint' to check for linting issues. These commands are essential for maintaining code quality.

When working with LangGraph components, follow established patterns: use StateGraph definitions, implement async functions for graph nodes, and extend BaseConfiguration for configuration classes.

For frontend development, follow Next.js 14 app router conventions and maintain TypeScript strict typing. Use the existing UI component patterns from the components/ui/ directory.

Always check existing utility functions in backend/utils.py and frontend/app/utils/ before creating new helper functions to avoid code duplication.
</general_rules>

<repository_structure>
The repository is organized into three main areas:

Backend (Python/LangGraph): Located in /backend/ directory containing the core retrieval and chat functionality. Key modules include configuration.py, retrieval.py, ingest.py, and utils.py. The main LangGraph implementation resides in /backend/retrieval_graph/ with graph.py defining the conversation flow, state.py managing state, and a nested /researcher_graph/ submodule for research operations.

Frontend (Next.js/TypeScript): Located in /frontend/app/ following Next.js 14 app router structure. Contains components/, contexts/, hooks/, utils/ directories, and API routes in api/. The main entry points are page.tsx and layout.tsx.

Scripts and Configuration: Utility scripts for evaluation and data processing are in /_scripts/. Root-level configuration includes langgraph.json for LangGraph Cloud deployment, pyproject.toml for Python dependencies, and frontend/package.json for Node.js dependencies.

The backend uses a modular approach where retrieval_graph/ contains the core conversational AI logic, while the main backend/ modules handle data ingestion, embeddings, and utility functions.
</repository_structure>

<dependencies_and_installation>
Backend dependencies are managed with Poetry. Run 'poetry install --with dev' to install all dependencies including development tools. Python 3.11+ is required.

Frontend dependencies use Yarn as the package manager. Navigate to the frontend/ directory and run 'yarn install' to install all Node.js dependencies.

Key backend dependencies include the LangChain ecosystem (langchain, langchain-core, langchain-community, langchain-openai, langchain-anthropic, langchain-weaviate), LangGraph (≥0.4.5), and Weaviate client (^4.0.0) for vector storage.

Key frontend dependencies include Next.js (14.2.25), React (18.2.0), TypeScript (5.1.6), and the LangChain SDK (@langchain/langgraph-sdk) for API communication.

Development tools include ruff for Python linting/formatting and prettier for frontend code formatting. These are automatically installed with the --with dev flag for Poetry and included in frontend devDependencies.
</dependencies_and_installation>

<testing_instructions>
Backend testing uses pytest framework focusing on end-to-end evaluation tests located in backend/tests/evals/. The primary test file is test_e2e.py which evaluates the entire retrieval and answer generation pipeline.

Run backend tests with 'poetry run pytest backend/tests/evals'. Tests focus on three key metrics: retrieval recall (≥0.65), answer correctness (≥0.9), and answer vs context correctness (≥0.9).

Testing requires several API keys as environment variables: LANGSMITH_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, WEAVIATE_URL, and WEAVIATE_API_KEY. These are used for evaluation against LangSmith datasets.

The evaluation tests use the "chat-langchain-qa" dataset from LangSmith and employ the aevaluate function for async evaluation with a judge model (Claude 3.5 Haiku) to score answer quality.

Tests are designed as regression tests to ensure the system maintains quality thresholds rather than unit testing individual components. Focus is on end-to-end functionality and answer quality metrics.

Frontend testing is not extensively implemented in the current codebase, with the focus being on backend evaluation and quality assurance.
</testing_instructions>

