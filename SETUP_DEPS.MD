## 🔧 **Step 1: Complete Environment Variables**

Create/update your `.env` file with **all required variables**:

```bash
# ✅ Already working - PostgreSQL
RECORD_MANAGER_DB_URL=postgresql://postgres:zkdtn1234@localhost:5432/chat_langchain

# 🔑 Required - OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# 🗄️ Required - Weaviate Vector Store
WEAVIATE_URL=your_weaviate_cluster_url
WEAVIATE_API_KEY=your_weaviate_api_key

# 📊 Optional - LangSmith (for monitoring/tracing)
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=chat-langchain

# 🔄 Optional - Ingestion settings
FORCE_UPDATE=false

# 🤖 Optional - Additional AI providers
ANTHROPIC_API_KEY=your_anthropic_key
FIREWORKS_API_KEY=your_fireworks_key
```

## 🚀 **Step 2: Set Up Weaviate (Vector Store)**

**Option A: Weaviate Cloud (Recommended)**

1. Go to [Weaviate Cloud Console](https://console.weaviate.cloud/)
2. Create account → Create Cluster
3. Get your cluster URL and API key
4. Add to `.env` file

**Option B: Local Weaviate (Docker)**

```yaml
# Add to your docker-compose.yml
  weaviate:
    image: cr.weaviate.io/semitechnologies/weaviate:1.25.0
    container_name: chat_langchain_weaviate
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
    volumes:
      - weaviate_data:/var/lib/weaviate

# Then add to volumes:
volumes:
  weaviate_data:
```

If using local Weaviate:

```bash
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=  # Leave empty for local
```

## 📦 **Step 3: Install Dependencies**

```bash
# Make sure you're in project root
cd /Users/MM7V26R293/Documents/Naver/AI-MCP/Langchain-practice/chat-langchain-practice

# Install Python dependencies
poetry install

# Verify installation
poetry run python -c "import langchain, langgraph; print('✅ Dependencies installed!')"
```

## 🧪 **Step 4: Test Core Components**

```bash
# Test environment setup
poetry run python -c "
import os
from backend.configuration import BaseConfiguration
from backend.embeddings import get_embeddings_model
print('✅ Configuration loaded!')
"

# Test database connection (you already did this)
poetry run python -c "
from langchain.indexes import SQLRecordManager
rm = SQLRecordManager('test', db_url='postgresql://postgres:zkdtn1234@localhost:5432/chat_langchain')
rm.create_schema()
print('✅ Database connection working!')
"
```

## 🏃‍♂️ **Step 5: Run the Server**

**Option A: LangGraph Test (Stateless)**

```bash
# Start PostgreSQL if not running
docker-compose up -d postgres

# Run LangGraph in test mode
poetry run langgraph test

# This should start the server on http://localhost:8123
```

**Option B: LangGraph Full Server (if you have license)**

```bash
poetry run langgraph up
```

## 📊 **Step 6: Populate Vector Store (Optional)**

If you want to test with actual data:

```bash
# Run document ingestion
poetry run python backend/ingest.py
```

## ✅ **Quick Setup Checklist**

- [x] PostgreSQL running ✅
- [ ] Get OpenAI API key
- [ ] Set up Weaviate (cloud or local)
- [ ] Create complete `.env` file
- [ ] Run `poetry install`
- [ ] Test with `langgraph test`
- [ ] Access at `http://localhost:8123`

## 🚨 **Priority Next Steps**

1. **Get OpenAI API key** - This is required for embeddings
2. **Set up Weaviate** - This stores your document vectors
3. **Complete `.env` file** with all variables
4. **Run `langgraph test`** to start the server

The server should start successfully once you have the OpenAI API key and Weaviate configured! Let me know when you get to Step 5 and I can help troubleshoot any issues.
