# Document Ingestion Flow - Complete Guide

A comprehensive guide to understanding how Chat LangChain ingests, processes, and stores documents for semantic search.

## 🎯 What is Document Ingestion?

Document ingestion is the process of taking raw documents (like web pages), converting them into searchable vectors, and storing them in a database. This enables AI assistants to search through documents by **meaning** rather than just keywords.

Think of it like creating an index for a library - but instead of organizing by title or author, we organize by the **meaning** of the content.

## 📦 Key Components

### 1. Vector Database (Weaviate)

**Purpose:** Store document content and vectors for semantic search

**What it stores:**

- **Vectors**: Arrays of 1,536 numbers representing document meaning
- **Text**: The actual document content
- **Metadata**: Source URL, title, description

**Used for:**

- Searching documents by meaning
- Retrieving relevant content for user queries
- Powering the AI chat responses

### 2. Record Manager (PostgreSQL)

**Purpose:** Track which documents have been indexed to prevent duplicates

**What it stores:**

- **Document Key**: Unique identifier (usually the URL)
- **Content Hash**: Fingerprint of the document content
- **Timestamp**: When it was last indexed

**Used for:**

- Detecting if a document already exists
- Detecting if a document's content changed
- Preventing duplicate processing and costs

### 3. Embedding Model (OpenAI)

**Purpose:** Convert text into numerical vectors

**What it does:**

- Takes text as input
- Returns an array of 1,536 floating-point numbers
- Same text always produces the same vector
- Similar texts produce similar vectors

## 🔄 Complete Ingestion Flow

```mermaid
flowchart TD
    A[📄 Load Documents from Web] --> B[🔪 Split into Chunks]
    B --> C[📋 Extract Metadata]
    C --> D{For Each Chunk}

    D --> E[📝 Generate Document Key]
    E --> F[🔐 Generate Content Hash]
    F --> G{Check Record Manager<br/>Does Key Exist?}

    G -->|No - New Document| H[🧠 Call Embedding API]
    H --> I[💾 Insert to Vector DB]
    I --> J[✅ Record in Record Manager]
    J --> K[Status: ADDED]

    G -->|Yes - Exists| L{Compare Content Hash<br/>Has Content Changed?}

    L -->|No - Same| M[⏭️ Skip Processing]
    M --> N[Status: SKIPPED]

    L -->|Yes - Changed| O[🧠 Call Embedding API]
    O --> P[🔄 Update Vector DB]
    P --> Q[🔄 Update Record Manager]
    Q --> R[Status: UPDATED]

    K --> S[📊 Return Stats]
    N --> S
    R --> S

    style H fill:#90EE90
    style O fill:#90EE90
    style M fill:#FFD700
    style I fill:#87CEEB
    style P fill:#87CEEB
```

## 📖 Step-by-Step Explanation

### Step 1: Load Documents from Web

```python
# Load from sitemaps
langchain_python_docs = load_langchain_python_docs()
langchain_js_docs = load_langchain_js_docs()
aggregated_site_docs = load_aggregated_docs_site()

# Example: 1 document loaded (4000 characters)
doc = {
    "page_content": "LangChain is a framework for developing...",
    "metadata": {
        "source": "https://python.langchain.com/docs/intro",
        "title": "Introduction to LangChain"
    }
}
```

### Step 2: Split Documents into Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,      # Max 4000 characters per chunk
    chunk_overlap=200     # 200 characters overlap between chunks
)

# Example: 1 document (4000 chars) → 1 chunk
# Larger doc (10000 chars) → 3 chunks with overlap
```

**Why split documents?**

- Embedding models have size limits
- Smaller chunks = more precise search results
- Overlap ensures context isn't lost between chunks

### Step 3: Generate Embeddings (Convert to Vectors)

```python
# Input: Text chunk
text = "LangChain is a framework for developing applications..."

# Process: Send to OpenAI API
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
vector = embedding_model.embed_documents([text])

# Output: Array of 1,536 numbers
vector = [0.023, -0.015, 0.034, 0.001, -0.019, ..., 0.008]
```

**What is a vector?**

- A mathematical representation of text meaning
- 1,536 floating-point numbers (for text-embedding-3-small)
- Similar texts have similar vectors
- Enables "meaning-based" search

### Step 4: Check Record Manager

```python
# Extract document key (unique identifier)
doc_key = doc.metadata["source"]  # "https://python.langchain.com/docs/intro"

# Generate content hash
content_hash = hash(doc.page_content + doc_key)  # "a3f5c9b2e8d1..."

# Check if already indexed
record = record_manager.get(key=doc_key)
```

**Three possible outcomes:**

#### A. New Document (Not Found)

```python
# Record doesn't exist
if record is None:
    # → Generate embedding
    # → Insert to vector DB
    # → Record in Record Manager
    status = "ADDED"
```

#### B. Unchanged Document (Hash Matches)

```python
# Record exists, hash matches
if record.content_hash == content_hash:
    # → Skip! No processing needed
    # → Save API costs and time
    status = "SKIPPED"
```

#### C. Updated Document (Hash Different)

```python
# Record exists, but content changed
if record.content_hash != content_hash:
    # → Generate new embedding
    # → Update vector DB
    # → Update Record Manager
    status = "UPDATED"
```

### Step 5: Store in Vector Database

```python
# What gets stored in Weaviate
{
    "_id": "uuid-abc-123",
    "vector": [0.023, -0.015, 0.034, ..., 0.008],  # 1,536 numbers
    "text": "LangChain is a framework for developing...",
    "source": "https://python.langchain.com/docs/intro",
    "title": "Introduction to LangChain"
}
```

### Step 6: Record in Record Manager

```python
# What gets stored in PostgreSQL
{
    "key": "https://python.langchain.com/docs/intro",
    "content_hash": "a3f5c9b2e8d1a4f7c3e9",
    "updated_at": "2024-10-25 10:30:00",
    "namespace": "weaviate/LangChain_general_guides"
}
```

## 🔍 How Search Works After Ingestion

```mermaid
flowchart LR
    A[👤 User Query:<br/>'How does memory work?'] --> B[🧠 Convert Query to Vector]
    B --> C[📊 Calculate Similarity<br/>with All Stored Vectors]
    C --> D[🎯 Find Closest Matches]
    D --> E[📄 Return Matching Documents]
    E --> F[🤖 Send to LLM for Answer]

    style B fill:#90EE90
    style C fill:#FFD700
    style E fill:#87CEEB
```

**Example:**

```python
# User asks
user_query = "How does memory work in LangChain?"

# Convert to vector
query_vector = [0.019, -0.013, 0.036, ..., 0.007]

# Weaviate calculates similarity
# Vector about memory: similarity = 0.89 (high!)
# Vector about chains: similarity = 0.45 (low)

# Returns most similar documents
results = [
    {
        "text": "Memory allows your applications to store and recall...",
        "source": "https://python.langchain.com/docs/memory",
        "similarity": 0.89
    }
]
```

## 📊 Real-World Example with Numbers

### Scenario: Ingesting LangChain Documentation

**First Run (Initial Load):**

```python
# Documents loaded: 1,500
# After splitting: 7,500 chunks
# Embeddings generated: 7,500
# API cost: ~$15
# Time: ~15 minutes

indexing_stats = {
    'num_added': 7500,
    'num_updated': 0,
    'num_skipped': 0,
    'num_deleted': 0
}
```

**Second Run (No Changes):**

```python
# Documents loaded: 1,500 (same)
# After splitting: 7,500 chunks (same)
# Embeddings generated: 0 ✅ (all skipped!)
# API cost: ~$0.10
# Time: ~30 seconds

indexing_stats = {
    'num_added': 0,
    'num_updated': 0,
    'num_skipped': 7500,  # All documents unchanged
    'num_deleted': 0
}
```

**Third Run (100 New Docs, 50 Updated):**

```python
# New documents: 100 → 500 chunks
# Updated documents: 50 → 200 chunks
# Unchanged: 1,350 → 6,800 chunks
# Embeddings generated: 700 (only new + updated)
# API cost: ~$1.50
# Time: ~2 minutes

indexing_stats = {
    'num_added': 500,      # New chunks
    'num_updated': 200,    # Changed chunks
    'num_skipped': 6800,   # Unchanged chunks
    'num_deleted': 0
}
```

## 💡 Why This System is Smart

### 1. **Cost Efficiency**

- **Without Record Manager:** Re-process everything every time = $15 per run
- **With Record Manager:** Only process changes = $1.50 per run
- **Savings:** 90% cost reduction! 💰

### 2. **No Duplicates**

- **Without Record Manager:** Each run adds duplicates
  - Run 1: 7,500 vectors
  - Run 2: 15,000 vectors (50% duplicates!)
  - Run 3: 22,500 vectors (67% duplicates!)
- **With Record Manager:** Always exactly the right number
  - Run 1: 7,500 vectors
  - Run 2: 7,500 vectors (same)
  - Run 3: 7,500 vectors (same)

### 3. **Speed**

- Skip unchanged documents
- Only generate embeddings when needed
- 85% faster on subsequent runs

### 4. **Automatic Cleanup**

Using `cleanup="full"` removes deleted documents:

```python
# If a page is removed from the source website
# Record Manager detects: "This was indexed before but not in current batch"
# → Automatically deletes from Vector DB
# → Removes from Record Manager
# → Keeps database clean and up-to-date
```

## 🔧 Implementation in Code

```python
def ingest_docs():
    # 1. Load and prepare documents
    docs = ingest_general_guides_and_tutorials()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=200
    )
    docs_transformed = text_splitter.split_documents(docs)

    # 2. Initialize embedding model
    embedding = get_embeddings_model()

    # 3. Connect to Vector DB
    vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        index_name="LangChain_general_guides",
        embedding=embedding
    )

    # 4. Initialize Record Manager
    record_manager = SQLRecordManager(
        "weaviate/LangChain_general_guides",
        db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    # 5. Index with smart deduplication
    indexing_stats = index(
        docs_transformed,           # Documents to process
        record_manager,             # Tracks what's indexed
        vectorstore,                # Where to store vectors
        cleanup="full",             # Remove deleted docs
        source_id_key="source"      # Use 'source' URL as key
    )

    print(f"Indexing complete: {indexing_stats}")
```

## 🎯 Key Takeaways

1. **Two Databases, Two Purposes:**

   - Vector DB (Weaviate) = Search engine for finding documents
   - Record Manager (PostgreSQL) = Bookkeeper preventing duplicates

2. **Smart Processing:**

   - Only generate embeddings for new/changed documents
   - Skip unchanged documents automatically
   - Clean up deleted documents

3. **Content Hash is Key:**

   - Document ID (URL) = Identity ("Who are you?")
   - Content Hash = Version ("Have you changed?")

4. **Production Ready:**
   - Cost efficient (90% savings)
   - Fast (85% faster on re-runs)
   - Reliable (no duplicates)
   - Automatic (handles updates and deletions)

## 📚 Quick Reference

### Vector Database Schema

```json
{
  "vector": [1536 floats],
  "text": "document content",
  "source": "url",
  "title": "page title"
}
```

### Record Manager Schema

```json
{
  "key": "document_url",
  "content_hash": "hash_of_content",
  "updated_at": "timestamp",
  "namespace": "database_namespace"
}
```

### Index Function Parameters

```python
index(
    docs,                    # Documents to process
    record_manager,          # Deduplication tracker
    vectorstore,             # Storage destination
    cleanup="full",          # "full" | "incremental" | None
    source_id_key="source",  # Metadata field for document ID
    force_update=False       # Force re-embedding everything
)
```

## 🚀 Running the Ingestion

```bash
# Set environment variables
export WEAVIATE_URL="your-weaviate-url"
export WEAVIATE_API_KEY="your-api-key"
export RECORD_MANAGER_DB_URL="postgresql://user:pass@host/db"
export OPENAI_API_KEY="your-openai-key"

# Run ingestion
python backend/ingest.py

# Output:
# INFO: Indexing stats: {'num_added': 7500, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
# INFO: General Guides and Tutorials now has this many vectors: 7500
```

---

**Built with:** LangChain, Weaviate, PostgreSQL, OpenAI Embeddings

**Last Updated:** October 25, 2024
