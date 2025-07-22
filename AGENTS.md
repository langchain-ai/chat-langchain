<general_rules>
- Always search existing functions and modules before creating new ones. Check backend/ directory for Python utilities, retrieval_graph/ for LangGraph components, and frontend/app/components/ for React components.
- Use ruff for Python code formatting and linting. Run `make format` to format code and `make lint` to check for issues before committing.
- Follow the established configuration pattern using dataclasses (BaseConfiguration, AgentConfiguration) when adding new configurable parameters.
- When adding new retrieval or LLM functionality, check backend/utils.py for existing helper functions like load_chat_model() and format_docs().
- For frontend development, use the existing UI component library in frontend/app/components/ui/ before creating new components.
- Always update type definitions in relevant TypeScript files when modifying frontend interfaces.
- Use structured outputs with Pydantic models for LLM interactions to ensure type safety and validation.
- Follow the existing prompt template patterns in backend/retrieval_graph/prompts.py when creating new prompts.
</general_rules>

<repository_structure>
- **Backend (backend/)**: Python/LangGraph application using Poetry for dependency management. Core modules include configuration.py for settings, retrieval_graph/ for LangGraph conversation flow, ingest.py for document processing, and retrieval.py for vector store operations.
- **Frontend (frontend/)**: Next.js 14 TypeScript application using Yarn package manager. Main structure includes app/ directory with page.tsx as entry point, components/ for React components, hooks/ for custom hooks, and contexts/ for state management.
- **LangGraph Architecture**: The retrieval_graph/ directory contains the conversational AI logic with graph.py as main orchestrator, researcher_graph/ for complex query handling, and state.py for conversation state management.
- **Configuration**: langgraph.json defines LangGraph Cloud deployment settings, pyproject.toml manages Python dependencies, and package.json handles Node.js dependencies.
- **Scripts**: _scripts/ directory contains utility scripts for evaluation and indexing operations.
- **Documentation**: Additional docs in CONCEPTS.md, MODIFY.md, LANGSMITH.md, PRODUCTION.md, and DEPLOYMENT.md provide detailed implementation guidance.
</repository_structure>

<dependencies_and_installation>
- **Backend Dependencies**: Use Poetry for Python dependency management. Run `poetry install --with dev` to install all dependencies including development tools. Python 3.11+ required.
- **Frontend Dependencies**: Use Yarn for Node.js dependency management. Run `yarn install` in the frontend/ directory to install all packages.
- **Environment Variables**: Copy .env.gcp.yaml.example to set up required API keys for OpenAI, Anthropic, Weaviate, and LangSmith services.
- **LangGraph Cloud**: This application is designed for LangGraph Cloud deployment. Local development requires LangGraph Cloud account or use the langserve branch for local development.
- **Vector Store**: Weaviate is used as the vector database. Ensure WEAVIATE_URL and WEAVIATE_API_KEY are configured in environment variables.
</dependencies_and_installation>

<testing_instructions>
- **Backend Testing**: Use pytest as the testing framework. Run `poetry run pytest backend/tests/evals` to execute end-to-end evaluation tests.
- **LangSmith Evaluations**: Tests in backend/tests/evals/ use LangSmith for evaluation metrics including retrieval recall, answer correctness, and context relevance. These tests require LANGSMITH_API_KEY environment variable.
- **Evaluation Metrics**: Tests validate retrieval quality (≥65% recall), answer correctness (≥90%), and context-answer alignment (≥90%) using automated scoring with Claude-3.5-Haiku as judge model.
- **CI/CD Testing**: GitHub workflows automatically run linting (.github/workflows/lint.yml) and evaluations (.github/workflows/eval.yml) on pull requests and pushes to master.
- **Frontend Testing**: Use Next.js built-in linting with `yarn lint` and formatting with `yarn format` using Prettier.
- **Manual Testing**: Test the full conversation flow including query routing, document retrieval, and response generation through the web interface.
</testing_instructions>

