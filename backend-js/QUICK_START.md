# Quick Start Guide

Get the JavaScript/TypeScript backend running in minutes!

## Prerequisites

- **Node.js 20+** installed (use nvm: `nvm use`)
- **pnpm** (enable with: `corepack enable`)
- Access to Weaviate instance
- API keys for LLM providers (OpenAI, Anthropic, Groq, or Ollama)
- LangSmith API key (for prompts and tracing)

## 1. Environment Setup

```bash
cd backend-js

# Copy environment template
cp env.example .env

# Edit .env with your credentials
nano .env
```

Required variables:

```bash
# At minimum, set these:
LANGCHAIN_API_KEY=your_langsmith_key
WEAVIATE_URL=your_weaviate_url
WEAVIATE_API_KEY=your_weaviate_key

# And at least one LLM provider:
GROQ_API_KEY=your_groq_key  # Recommended: fast and free tier
# OR
OPENAI_API_KEY=your_openai_key
# OR
ANTHROPIC_API_KEY=your_anthropic_key
```

## 2. Install Dependencies

```bash
# Enable Corepack (one-time setup)
corepack enable

# Install dependencies
pnpm install
```

## 3. Ingest Documents (First Time Only)

```bash
pnpm ingest
```

This will:

- Load documents from LangChain docs
- Split into chunks
- Generate embeddings
- Index in Weaviate

**Expected time:** 5-15 minutes (depending on internet speed)

## 4. Test the Graph

Create a test file:

```bash
# Create test.ts
cat > test.ts << 'EOF'
import { HumanMessage } from "@langchain/core/messages";
import { graph } from "./src/retrieval_graph/graph.js";

async function test() {
  const result = await graph.invoke({
    messages: [new HumanMessage("What is LangChain?")],
  });

  console.log("\n=== Answer ===");
  console.log(result.answer);
  console.log("\n=== Docs Retrieved ===");
  console.log(result.documents.length);
}

test();
EOF

# Run test
npx tsx test.ts
```

**Expected output:** An answer about LangChain with several documents retrieved.

## 5. Start Self-Hosted Server

```bash
pnpm dev
```

**Expected output:**

```
╔═══════════════════════════════════════════════════════════╗
║   🚀 Chat LangChain Backend Server (Self-Hosted)        ║
║   Status: Running                                         ║
║   Port:   3001                                            ║
║   URL:    http://localhost:3001                          ║
╚═══════════════════════════════════════════════════════════╝
```

## 6. Test the API

```bash
# In another terminal:
curl -X POST http://localhost:3001/runs \
  -H "Content-Type: application/json" \
  -d '{"messages": ["What is LangChain?"]}'
```

**Expected:** JSON response with answer and documents.

## 7. Deploy to LangGraph Cloud (Optional)

```bash
# Make sure langgraph CLI is installed
npm install -g langgraph-cli

# Deploy
langgraph deploy
```

Follow the prompts to complete deployment.

## Common Issues

### "Cannot find module"

**Solution:** Ensure imports use `.js` extension:

```typescript
import { something } from './module.js' // ✅ Correct
import { something } from './module' // ❌ Wrong
```

### "Weaviate connection failed"

**Solution:**

1. Check WEAVIATE_URL is correct
2. Verify API key is valid
3. Test with: `curl https://your-weaviate-url/v1/meta`

### "No documents retrieved"

**Solution:** Run ingestion: `pnpm ingest`

### "API key missing"

**Solution:** Check `.env` file has all required keys

## Next Steps

1. ✅ Test with different questions
2. ✅ Run evaluations: `pnpm test:e2e`
3. ✅ Read `TESTING_GUIDE.md` for comprehensive testing
4. ✅ Read `FRONTEND_INTEGRATION.md` for frontend setup
5. ✅ Read `MIGRATION_SUMMARY.md` for detailed information

## Need Help?

- Check `README.md` for detailed documentation
- Review `TESTING_GUIDE.md` for debugging tips
- See `MIGRATION_SUMMARY.md` for architecture details
- Create an issue on GitHub with full error details

## Success Checklist

- [ ] Environment variables configured
- [ ] Dependencies installed
- [ ] Documents ingested
- [ ] Test graph runs successfully
- [ ] Server starts without errors
- [ ] API responds to requests
- [ ] (Optional) Deployed to LangGraph Cloud

**Congratulations!** Your JavaScript backend is now running. 🎉
