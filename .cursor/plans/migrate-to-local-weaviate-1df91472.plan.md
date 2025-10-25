<!-- 1df91472-d73d-47a7-8bf6-a4f0f3c34bca 761d6e0b-bf93-4fb8-acbc-c61e0c924bfb -->
# Migrate to Local Weaviate with Transformers

## Overview

Replace OpenAI embeddings and Weaviate Cloud with a local Weaviate Docker instance using the `text2vec-transformers` module (`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`) for cost-free embeddings.

## Key Changes

### 1. Update Docker Infrastructure

**File: `docker-compose.yml`**

Add Weaviate service with text2vec-transformers module:

- Weaviate core service on port 8080
- text2vec-transformers service with `sentence-transformers-multi-qa-MiniLM-L6-cos-v1` model
- CPU-only configuration (ENABLE_CUDA: 0)
- Persistent volume for data storage

### 2. Update Weaviate Connection Logic

**File: `backend/ingest.py`**

Changes:

- Replace `weaviate.connect_to_weaviate_cloud()` with `weaviate.connect_to_local()`
- Remove `WEAVIATE_API_KEY` usage (local instance doesn't need auth)
- Configure WeaviateVectorStore to use Weaviate's built-in vectorizer instead of external embeddings
- The `text2vec-transformers` module will handle vectorization automatically

**File: `backend/retrieval.py`**

Changes:

- Update `make_weaviate_retriever()` to connect to local instance
- Modify `make_text_encoder()` to support `weaviate/text2vec-transformers` option
- Handle embedding model configuration for local vs cloud scenarios

### 3. Update Embedding Configuration

**File: `backend/embeddings.py`**

Update `get_embeddings_model()` to return `None` or a placeholder when using Weaviate's built-in vectorizer, since Weaviate handles embeddings internally.

**File: `backend/configuration.py`**

Update default `embedding_model` from `"openai/text-embedding-3-small"` to `"weaviate/text2vec-transformers"` to reflect the new local setup.

### 4. Environment Variables

Update `.env` file:

- Change `WEAVIATE_URL` from cloud URL to `http://localhost:8080`
- `WEAVIATE_API_KEY` can be removed or left empty (not needed for local)

### 5. Optional: Update Index Names

**File: `backend/constants.py`**

Consider renaming index names to reflect the new embedding model (e.g., replace `OpenAI_text_embedding_3_small` suffix with `transformers_multi_qa_MiniLM`).

## Technical Notes

- Weaviate's text2vec-transformers module vectorizes text automatically at ingestion and query time
- No external API calls for embeddings = zero cost
- CPU-only transformer inference will be slower than GPU but functional
- First run will download the transformer model (~80MB)
- Data persists in Docker volume between restarts

### To-dos

- [ ] Add Weaviate and text2vec-transformers services to docker-compose.yml with proper configuration
- [ ] Modify backend/ingest.py to connect to local Weaviate and remove external embedding dependency
- [ ] Modify backend/retrieval.py to support local Weaviate connection and built-in vectorizer
- [ ] Update backend/embeddings.py and backend/configuration.py for Weaviate built-in vectorizer
- [ ] Start Docker services and test ingestion with local Weaviate to verify embeddings work correctly