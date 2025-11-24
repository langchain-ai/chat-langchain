# Chat LangChain - JavaScript/TypeScript Backend

This is the JavaScript/TypeScript implementation of the chat-langchain backend, migrated from Python to leverage LangChain.js v1 and modern JavaScript tooling.

## Architecture

Built with:

- **LangChain.js v1.0** - Modern LangChain with standard content blocks
- **@langchain/core v1.0.5** - Core abstractions and utilities
- **LangGraph.js v0.2** - StateGraph pattern for complex agent workflows
- **TypeScript 5.6** - Full type safety and excellent DX
- **Weaviate** - Vector store for document retrieval
- **Vitest** - Fast testing framework
- **pnpm** - Fast, disk space efficient package manager

## Project Structure

```
backend-js/
├── src/
│   ├── retrieval_graph/       # Main agent graph
│   │   ├── graph.ts           # Main retrieval graph
│   │   ├── state.ts           # State management
│   │   ├── configuration.ts   # Configuration schemas
│   │   ├── prompts.ts         # LangSmith prompts
│   │   └── researcher_graph/  # Researcher subgraph
│   ├── retrieval.ts           # Retrieval logic
│   ├── embeddings.ts          # Embedding models
│   ├── utils.ts               # Utility functions
│   ├── constants.ts           # Constants
│   ├── ingest.ts              # Document ingestion
│   ├── parser.ts              # HTML parsing
│   └── server.ts              # Self-hosted Express server
├── tests/
│   └── evals/                 # Evaluation tests
├── scripts/                   # Helper scripts
└── package.json
```

## Setup

### 1. Prerequisites

This project requires:

- **Node.js 20+** (use nvm: `nvm use` to load from `.nvmrc`)
- **pnpm** (specified in `package.json`)

### 2. Install Dependencies

```bash
cd backend-js

# Enable Corepack for pnpm (one-time setup)
corepack enable

# Install dependencies
pnpm install
```

### 2. Configure Environment

Copy `env.example` to `.env` and fill in your API keys:

```bash
cp env.example .env
```

Required environment variables:

- `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` or `GROQ_API_KEY` - For LLM providers
- `LANGCHAIN_API_KEY` - For LangSmith tracing and prompts
- `WEAVIATE_URL` - Your Weaviate instance URL
- `WEAVIATE_API_KEY` - Weaviate authentication

### 3. Ingest Documents (First Time)

```bash
pnpm ingest
```

This will:

- Load documents from LangChain docs
- Split into chunks
- Generate embeddings
- Index in Weaviate

## Development

### Run Self-Hosted Server

```bash
pnpm dev
```

Server will be available at `http://localhost:3001`

### API Endpoints

- `POST /runs` - Invoke graph with new input
- `POST /runs/stream` - Stream graph execution (SSE)
- `GET /runs/:run_id` - Get run status
- `POST /threads/:thread_id/runs` - Continue conversation thread
- `GET /threads/:thread_id/state` - Get thread state

### Run Tests

```bash
# All tests
pnpm test

# E2E evaluation tests
pnpm test:e2e
```

### Type Checking

```bash
pnpm typecheck
```

### Build for Production

```bash
pnpm build
pnpm start
```

## Deployment

### Option 1: LangGraph Cloud (Recommended)

1. Ensure `langgraph.json` is configured:

```json
{
  "$schema": "https://langgra.ph/schema.json",
  "dependencies": ["."],
  "graphs": {
    "chat": "./src/retrieval_graph/graph.ts:graph"
  },
  "env": ".env"
}
```

2. Deploy:

```bash
langgraph deploy
```

Benefits:

- Managed checkpointing and state persistence
- Automatic streaming endpoints
- Thread management UI
- Automatic scaling
- Zero infrastructure maintenance

### Option 2: Self-Hosted

1. Set up PostgreSQL database for checkpointing
2. Configure `DATABASE_URL` in `.env`
3. Build and run:

```bash
pnpm build
pnpm start
```

Benefits:

- Full control over infrastructure
- Custom middleware and integrations
- Cost optimization
- No vendor lock-in

## Key Features

### LangGraph StateGraph Pattern

Uses the StateGraph pattern for complex multi-step workflows:

1. **Research Planning** - Generate research steps
2. **Query Generation** - Create diverse search queries
3. **Parallel Retrieval** - Fetch documents in parallel
4. **Response Generation** - Synthesize answer from retrieved docs

### Evaluation System

Comprehensive evaluation pipeline with:

- Retrieval recall metrics
- Answer correctness (vs reference)
- Context faithfulness
- LangSmith integration for tracking

Run evaluations:

```bash
pnpm test:e2e
```

### Multi-Provider Support

Supports multiple LLM and embedding providers:

- OpenAI (GPT-4, GPT-3.5, text-embedding-3)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Haiku)
- Groq (Llama, Mixtral)
- Ollama (Local models)

## Comparison with Python Version

Both implementations coexist and share:

- Same Weaviate vector store
- Same evaluation datasets
- Same LangSmith prompts
- Same document corpus

This allows for:

- Performance comparison
- Feature parity validation
- Gradual migration
- A/B testing

## Migration Notes

This implementation maintains the same architecture as the Python version:

- StateGraph pattern (not using new `createAgent` API)
- Same node structure and conditional edges
- Same prompts and system messages
- Compatible API surface for frontend

See `../docs/` for detailed migration documentation.

## Contributing

This is a learning project. The Python implementation remains in `../backend/` directory.

## License

MIT
