# Document Ingestion Flow - Complete Guide

A comprehensive guide to understanding how Chat LangChain ingests, processes, and stores documents for semantic search using local Weaviate with Docker.

## 🎯 What is Document Ingestion?

Document ingestion converts raw documents (web pages) into searchable vectors stored in a database. This enables AI assistants to search by **meaning** rather than keywords.

## 🐳 Docker Infrastructure

### Three-Container Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │   PostgreSQL     │         │    Weaviate      │          │
│  │   :5432          │         │    :8080         │          │
│  │                  │         │    :50051        │          │
│  │  Record Manager  │         │  Vector Store    │          │
│  │  (Deduplication) │         │                  │          │
│  └──────────────────┘         └──────────────────┘          │
│                                                               │
│                          ┌──────────────────────┐            │
│                          │      Ollama          │            │
│                          │      :11434          │            │
│                          │                      │            │
│                          │  nomic-embed-text    │            │
│                          │  (2K context, 768d)  │            │
│                          └──────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### Container Roles

**1. PostgreSQL** (`:5432`)

- Tracks indexed documents for deduplication
- Stores content hashes and timestamps
- Prevents duplicate processing

**2. Weaviate** (`:8080`)

- Vector database server
- Stores vectors + text + metadata
- No built-in vectorizer (uses external embeddings)

**3. Ollama** (`:11434`)

- Embedding model: `nomic-embed-text`
- Converts text → 768-dimensional vectors
- 2K token context window
- Called directly by Python via LangChain

### Internal Communication Flow

```
Your Python Code
    │ "Store: 'What is LangChain?'"
    ↓
Ollama Server (:11434)
    │ Processes with nomic-embed-text
    ↓
Your Python Code
    │ Receives 768-dim vector [0.123, -0.456, ...]
    ↓
Weaviate Server (:8080)
    │ Stores: {text, vector, metadata}
    ↓
Your Python Code
    │ Response: "Document stored"
```

**Key Point:** Your Python code calls Ollama at `localhost:11434` for embeddings, then sends the vectors to Weaviate at `localhost:8080` for storage.

---

## 🔄 High-Level System Flow

### Ingestion Process

```
[Web Scraping]
    │ Load 32 pages from js.langchain.com
    ↓
[Text Splitting]
    │ Split into ~120 chunks (4000 chars each)
    ↓
[Python: ingest.py]
    │ Send text to Ollama
    ↓
[Ollama Server]
    │ Generate 768-dim vectors with nomic-embed-text
    ↓
[Python: ingest.py]
    │ Send vectors + text to Weaviate
    ↓
[Weaviate Server]
    │ Store: {text, vector, source, title}
    ↓
[Record Manager (PostgreSQL)]
    │ Track: source URL + content hash
    └─► Deduplication on next run
```

### Retrieval Process

```
[User Query]
    │ "What is LangChain?"
    ↓
[Python: retrieval.py]
    │ retriever.invoke(query)
    ↓
[Ollama Server]
    │ Generate query vector with nomic-embed-text
    ↓
[Python: retrieval.py]
    │ Send query vector to Weaviate
    ↓
[Weaviate Server]
    │ Find similar vectors (cosine similarity)
    │ Return top-k documents with distance scores
    ↓
[Python: retrieval.py]
    └─► List[Document] with metadata
```

---

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
```

**What does `load_langchain_js_docs()` return?**

A list of LangChain `Document` objects loaded from the sitemap:

```python
[
    Document(
        page_content="LangChain.js\n\nIntroduction\n\nLangChain is a framework for developing applications powered by language models...",
        metadata={
            "source": "https://js.langchain.com/docs/get_started/introduction",
            "title": "Introduction | 🦜️🔗 LangChain.js",
            "description": "LangChain is a framework for developing applications powered by language models.",
            "language": "en",
            "loc": "https://js.langchain.com/docs/get_started/introduction",
            "lastmod": "2024-10-15T10:30:00+00:00",
            "changefreq": "weekly",
            "priority": 0.8
        }
    ),
    Document(
        page_content="Installation\n\nTo get started with LangChain.js, install the package...",
        metadata={
            "source": "https://js.langchain.com/docs/get_started/installation",
            "title": "Installation | 🦜️🔗 LangChain.js",
            ...
        }
    ),
    # ... hundreds more documents from the sitemap
]
```

**Key Details:**

- **Return Type**: `List[Document]` (typically 100-500 documents)
- **Filtering**: Only URLs matching `https://js.langchain.com/docs/` are included
- **Content**: Clean text extracted via `simple_extractor()` (HTML tags removed, excessive newlines collapsed)
- **Metadata**: Source URL, title, description, language from HTML elements

### Step 2: Split Documents into Chunks

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000,      # Max 4000 characters per chunk
    chunk_overlap=400     # 400 characters overlap between chunks
)

docs_transformed = text_splitter.split_documents(general_guides_and_tutorials_docs)

# Filter out tiny chunks
docs_transformed = [doc for doc in docs_transformed if len(doc.page_content) > 10]
```

**Transformation Process:**

```mermaid
flowchart TD
    A[📄 Input: 1 Large Document<br/>8000 characters] --> B[🔪 RecursiveCharacterTextSplitter<br/>chunk_size=4000, overlap=200]
    B --> C[📋 Chunk 1<br/>chars 0-4000<br/>Same metadata]
    B --> D[📋 Chunk 2<br/>chars 3800-7800<br/>200 char overlap<br/>Same metadata]
    B --> E[📋 Chunk 3<br/>chars 7600-8000<br/>200 char overlap<br/>Same metadata]

    C --> F[✅ Filter: len > 10]
    D --> F
    E --> F

    F --> G[📦 Output: 3 Chunks<br/>Ready for embedding]

    style A fill:#FFE4B5
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#90EE90
    style G fill:#87CEEB
```

**Before Splitting:**

```python
# 1 document (8000 characters)
Document(
    page_content="LangChain.js\n\nIntroduction\n\nLangChain is a framework..." (8000 chars),
    metadata={
        "source": "https://js.langchain.com/docs/intro",
        "title": "Introduction | 🦜️🔗 LangChain.js"
    }
)
```

**After Splitting:**

```python
# Becomes 3 smaller chunks
[
    Document(
        page_content="LangChain.js\n\nIntroduction\n\nLangChain is a framework..." (4000 chars),
        metadata={
            "source": "https://js.langchain.com/docs/intro",  # Same!
            "title": "Introduction | 🦜️🔗 LangChain.js"      # Same!
        }
    ),
    Document(
        page_content="...powered by language models\n\nThe main value props..." (4000 chars, 200 overlap),
        metadata={
            "source": "https://js.langchain.com/docs/intro",  # Same!
            "title": "Introduction | 🦜️🔗 LangChain.js"      # Same!
        }
    ),
    Document(
        page_content="...for accomplishing specific tasks..." (400 chars, 200 overlap),
        metadata={
            "source": "https://js.langchain.com/docs/intro",  # Same!
            "title": "Introduction | 🦜️🔗 LangChain.js"      # Same!
        }
    )
]
```

**Key Points:**

- **Input**: ~100-500 large documents (potentially 5000+ chars each)
- **Output**: ~500-2000+ smaller chunks (max 4000 chars each)
- **Metadata preserved**: All chunks keep the same `source` URL (critical for record management!)
- **200-char overlap**: Ensures context isn't lost at boundaries

**Why split documents?**

- Embedding models have size limits
- Smaller chunks = more precise search results
- Overlap ensures context isn't lost between chunks

### Step 3: Generate Embeddings (Server-Side)

```python
# Python code generates embeddings via Ollama, then stores in Weaviate
vectorstore.add_documents([
    Document(page_content="LangChain is a framework...", metadata={...})
])

# What happens internally:
# 1. Python calls Ollama at localhost:11434
# 2. Ollama/nomic-embed-text returns 768-dim vector [0.023, -0.015, ..., 0.008]
# 3. Python sends {text, vector, metadata} to Weaviate
# 4. Weaviate stores the complete document
```

**What is a vector?**

- Mathematical representation of text meaning
- 768 floating-point numbers (nomic-embed-text)
- Similar texts have similar vectors
- Enables "meaning-based" search
- 2K token context window for better understanding
- **Generated locally via Ollama** (no API costs!)

### Step 4: Check Record Manager & Smart Deduplication

The Record Manager uses `source_id_key="source"` to **group chunks by their source URL** and track changes.

```python
# For each chunk after splitting
for doc in docs_transformed:
    # Extract document key (unique identifier)
    doc_key = doc.metadata["source"]  # e.g., "https://js.langchain.com/docs/intro"

    # Generate content hash
    content_hash = hash(doc.page_content + doc_key)  # e.g., "a3f5c9b2e8d1..."

    # Check if already indexed
    record = record_manager.get(key=doc_key)
```

**How Record Management Works:**

```mermaid
flowchart TD
    A[📄 Run 1: Load URL<br/>example.com/page1] --> B[🔪 Split into 3 Chunks]
    B --> C[📝 Chunk 1: hash_h1<br/>source: page1]
    B --> D[📝 Chunk 2: hash_h2<br/>source: page1]
    B --> E[📝 Chunk 3: hash_h3<br/>source: page1]

    C --> F[💾 Record Manager<br/>Groups by source_id_key]
    D --> F
    E --> F

    F --> G[📊 Stored Record<br/>source: page1<br/>hashes: h1, h2, h3<br/>vector_ids: v1, v2, v3]

    G --> H[📄 Run 2: Same URL<br/>Content Updated!]

    H --> I[🔪 Split into 3 New Chunks]
    I --> J[📝 Chunk 1: hash_h1_v2<br/>source: page1]
    I --> K[📝 Chunk 2: hash_h2_v2<br/>source: page1]
    I --> L[📝 Chunk 3: hash_h3_v2<br/>source: page1]

    J --> M{🔍 Compare by source: page1<br/>Old: h1, h2, h3<br/>New: h1_v2, h2_v2, h3_v2}
    K --> M
    L --> M

    M --> N[❌ Delete old vectors<br/>v1, v2, v3]
    M --> O[✅ Insert new vectors<br/>v1_v2, v2_v2, v3_v2]

    N --> P[📊 Updated Record<br/>source: page1<br/>hashes: h1_v2, h2_v2, h3_v2<br/>vector_ids: v1_v2, v2_v2, v3_v2]
    O --> P

    style G fill:#90EE90
    style M fill:#FFD700
    style N fill:#FF6B6B
    style O fill:#4ECDC4
    style P fill:#90EE90
```

**Real Example - First Run:**

```python
# URL: https://js.langchain.com/docs/intro (8000 chars)
# After splitting → 3 chunks

# Record Manager stores:
{
    "group_id": "https://js.langchain.com/docs/intro",  # source_id_key
    "doc_hashes": [
        "hash_abc123",  # Chunk 1 hash
        "hash_def456",  # Chunk 2 hash
        "hash_ghi789"   # Chunk 3 hash
    ],
    "vector_ids": ["vec_1", "vec_2", "vec_3"],
    "updated_at": "2024-10-25T10:00:00"
}
```

**Real Example - Second Run (Content Changed):**

```python
# Same URL, but website content updated
# After splitting → 3 NEW chunks (different content)

# Record Manager compares:
# - Groups by: "https://js.langchain.com/docs/intro"
# - Old hashes: [hash_abc123, hash_def456, hash_ghi789]
# - New hashes: [hash_xyz111, hash_xyz222, hash_xyz333]
# - Result: NO MATCHES!

# Actions:
# 1. DELETE old vectors: vec_1, vec_2, vec_3 from Weaviate
# 2. INSERT new vectors: vec_4, vec_5, vec_6 to Weaviate
# 3. UPDATE record:
{
    "group_id": "https://js.langchain.com/docs/intro",
    "doc_hashes": [
        "hash_xyz111",  # NEW Chunk 1 hash
        "hash_xyz222",  # NEW Chunk 2 hash
        "hash_xyz333"   # NEW Chunk 3 hash
    ],
    "vector_ids": ["vec_4", "vec_5", "vec_6"],  # NEW vector IDs
    "updated_at": "2024-10-26T14:30:00"
}
```

**Key Insight: Grouping by `source_id_key`**

The critical parameter is `source_id_key="source"` in the index function:

```python
indexing_stats = index(
    docs_transformed,
    record_manager,
    vectorstore,
    cleanup="full",
    source_id_key="source"  # ← Groups all chunks by their 'source' URL!
)
```

This means:

- All chunks from the **same URL are treated as a group**
- When content changes, **all old chunks are deleted** and **all new chunks are inserted**
- Record Manager tracks at the **source URL level**, not individual chunk level

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

```
User Query: "How does memory work?"
    │
    ↓
Python: retriever.invoke(query)
    │ Sends raw text to Weaviate
    ↓
Weaviate Server
    │ Calls text2vec-transformers:8080
    ↓
text2vec-transformers
    │ Returns query vector [0.019, -0.013, ...]
    ↓
Weaviate Server
    │ Calculates cosine similarity with all stored vectors
    │ Memory doc: distance = 0.12 (close!)
    │ Chains doc: distance = 0.89 (far)
    ↓
Python Code
    │ Receives top-k documents with distance scores
    └─► Returns: List[Document]
```

**Example:**

```python
from backend.retrieval import make_retriever

config = RunnableConfig(configurable={"search_kwargs": {"k": 3}})

with make_retriever(config) as retriever:
    # Uses Weaviate's near_text query (server-side vectorization)
    docs = retriever.invoke("How does memory work in LangChain?")

    for doc in docs:
        print(f"Title: {doc.metadata['title']}")
        print(f"Distance: {doc.metadata['distance']:.4f}")
        print(f"Content: {doc.page_content[:200]}...")
```

## 🎓 Record Management Summary

### How Chunks Are Tracked Across Runs

```mermaid
graph TB
    subgraph "Run 1: Initial Load"
        A1[URL: docs/intro<br/>Content: Version 1<br/>8000 chars] --> B1[Split]
        B1 --> C1[Chunk 1: h1]
        B1 --> C2[Chunk 2: h2]
        B1 --> C3[Chunk 3: h3]
        C1 --> D1[Record Manager]
        C2 --> D1
        C3 --> D1
        D1 --> E1[Store Group:<br/>source: docs/intro<br/>hashes: h1,h2,h3]
    end

    subgraph "Run 2: Content Changed"
        A2[URL: docs/intro<br/>Content: Version 2<br/>8000 chars] --> B2[Split]
        B2 --> C4[Chunk 1: h1v2]
        B2 --> C5[Chunk 2: h2v2]
        B2 --> C6[Chunk 3: h3v2]
        C4 --> D2[Record Manager]
        C5 --> D2
        C6 --> D2
        D2 --> E2{Compare by source:<br/>docs/intro}
        E2 --> F1[Old hashes:<br/>h1, h2, h3]
        E2 --> F2[New hashes:<br/>h1v2, h2v2, h3v2]
        F1 --> G[❌ DELETE old vectors]
        F2 --> H[✅ INSERT new vectors]
        G --> I[Update Record:<br/>source: docs/intro<br/>hashes: h1v2,h2v2,h3v2]
        H --> I
    end

    E1 -.->|Next run| A2

    style E1 fill:#90EE90
    style E2 fill:#FFD700
    style G fill:#FF6B6B
    style H fill:#4ECDC4
    style I fill:#90EE90
```

### Key Concepts

**1. One Source URL → Multiple Chunks → One Record Group**

```
URL: https://js.langchain.com/docs/intro
├── Chunk 1 (hash: abc123)
├── Chunk 2 (hash: def456)
└── Chunk 3 (hash: ghi789)

Record Manager Entry:
{
  group_id: "https://js.langchain.com/docs/intro",
  hashes: ["abc123", "def456", "ghi789"]
}
```

**2. Comparison Happens at Group Level**

- Record Manager groups all chunks by `source_id_key="source"`
- When the same URL is re-ingested:
  - Compare: Old hash list vs New hash list
  - If ANY hash is different → Update ALL chunks in that group
  - If all hashes match → Skip ALL chunks in that group

**3. `cleanup="full"` Removes Deleted Sources**

```python
# Run 1: 3 URLs
- docs/intro → 3 chunks
- docs/install → 2 chunks
- docs/guide → 4 chunks

# Run 2: docs/install removed from sitemap
- docs/intro → 3 chunks (unchanged, SKIP)
- docs/guide → 4 chunks (unchanged, SKIP)
# ❌ docs/install automatically DELETED from vector DB
```

**4. Time Savings Through Smart Tracking**

```
First Run:
- 32 URLs → 465 chunks
- Generate 465 embeddings (server-side)
- Time: ~8 seconds

Second Run (no changes):
- 32 URLs → 465 chunks
- Generate 0 embeddings (all skipped!)
- Time: ~1 second
- Savings: 87.5% time! ⚡

Third Run (5 URLs changed):
- 27 URLs unchanged → 390 chunks SKIPPED
- 5 URLs changed → 75 chunks UPDATED
- Generate 75 embeddings (server-side)
- Time: ~2 seconds
- Savings: 75% time! ⚡
```

## 📊 Real-World Example with Numbers

### Scenario: Ingesting LangChain JS Documentation

**First Run (Initial Load):**

```python
# Documents loaded: 32 pages (from sitemap)
# After splitting: ~120 chunks (4000 chars each)
# Embeddings generated: ~120 (via Ollama, no cost!)
# Time: ~10-15 seconds (local generation)

indexing_stats = {
    'num_added': 120,
    'num_updated': 0,
    'num_skipped': 0,
    'num_deleted': 0
}
```

**Second Run (No Changes):**

```python
# Documents loaded: 32 pages (same)
# After splitting: ~120 chunks (same)
# Embeddings generated: 0 ✅ (all skipped!)
# Time: ~1 second

indexing_stats = {
    'num_added': 0,
    'num_updated': 0,
    'num_skipped': 120,  # All documents unchanged
    'num_deleted': 0
}
```

**Third Run (5 Pages Updated):**

```python
# New documents: 0
# Updated documents: 5 pages → ~19 chunks
# Unchanged: 27 pages → ~101 chunks
# Embeddings generated: 19 (only updated chunks)
# Time: ~3 seconds

indexing_stats = {
    'num_added': 0,
    'num_updated': 19,     # Changed chunks
    'num_skipped': 101,    # Unchanged chunks
    'num_deleted': 0
}
```

## 💡 Why This System is Smart

### 1. **Local & Private**

- **All processing happens locally** in Docker containers
- Ollama runs embeddings on your machine
- Your data never leaves your machine
- No API costs for embeddings! 🎉
- 768-dimensional vectors for better semantic understanding

### 2. **No Duplicates**

- **Without Record Manager:** Each run adds duplicates
  - Run 1: 120 vectors
  - Run 2: 240 vectors (50% duplicates!)
  - Run 3: 360 vectors (67% duplicates!)
- **With Record Manager:** Always exactly the right number
  - Run 1: 120 vectors
  - Run 2: 120 vectors (same)
  - Run 3: 120 vectors (same)

### 3. **Speed & Efficiency**

- Skip unchanged documents (no re-processing)
- Only generate embeddings when content changes
- 85% faster on subsequent runs
- Larger chunks (4000 chars) = fewer vectors to process
- First run: ~10-15 seconds for 32 pages → ~120 vectors
- Second run (unchanged): ~1 second (all skipped)

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
    docs = ingest_general_guides_and_tutorials()  # 32 HTML pages
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,   # 4000 chars per chunk (larger for better context)
        chunk_overlap=400   # 400 chars overlap
    )
    docs_transformed = text_splitter.split_documents(docs)  # → ~120 chunks

    # 2. Get embedding model (OllamaEmbeddings for nomic-embed-text)
    embedding = get_embeddings_model()  # → OllamaEmbeddings

    # 3. Connect to local Weaviate
    with weaviate.connect_to_local() as weaviate_client:
        # 4. Create Vector Store (embedding=OllamaEmbeddings)
        vectorstore = WeaviateVectorStore(
            client=weaviate_client,
            index_name="LangChain_General_Guides_And_Tutorials_nomic_embed_text",
            text_key="text",
            embedding=embedding,  # ← Python handles vectorization via Ollama!
            attributes=["source", "title"]
        )

        # 5. Initialize Record Manager
        record_manager = SQLRecordManager(
            "weaviate/LangChain_General_Guides_...",
            db_url="postgresql://postgres:zkdtn1234@localhost:5432/chat_langchain"
        )
        record_manager.create_schema()

        # 6. Index with smart deduplication
        indexing_stats = index(
            docs_transformed,
            record_manager,
            vectorstore,
            cleanup="full",
            source_id_key="source",  # Group by URL
            force_update=False
        )

        print(f"Indexing complete: {indexing_stats}")
        # Output: {'num_added': 120, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
```

## 🎯 Key Takeaways

1. **Docker-Based Local Setup:**

   - **PostgreSQL** = Tracks indexed documents (deduplication)
   - **Weaviate** = Vector database server (storage only)
   - **Ollama** = Embedding service with nomic-embed-text (768-dim vectors, 2K context)
   - Python calls Ollama for embeddings, then stores in Weaviate

2. **Client-Side Vectorization via Ollama:**

   - Python calls Ollama at `localhost:11434`
   - Ollama generates vectors using nomic-embed-text
   - Python sends vectors to Weaviate for storage
   - No external API calls or costs
   - Your data stays private and local

3. **Document Pipeline:**

   - **Load**: 32 HTML pages from sitemap
   - **Split**: 32 pages → ~120 chunks (4000 chars, 400 overlap)
   - **Vectorize**: Via Ollama/nomic-embed-text (768 dimensions)
   - **Store**: Weaviate stores {text, vector, metadata}

4. **Smart Deduplication with Record Manager:**

   - Groups chunks by source URL (`source_id_key="source"`)
   - Tracks content hashes in PostgreSQL
   - **NEW**: First time seeing URL → insert all chunks
   - **SKIP**: URL exists, hashes match → skip (huge savings!)
   - **UPDATE**: URL exists, hash differs → delete old + insert new
   - **DELETE**: URL removed from source → auto-cleanup

5. **Production Benefits:**
   - **Private**: All local, no data leaves your machine
   - **Fast**: 85% faster on re-runs (skip unchanged)
   - **Reliable**: No duplicates, automatic cleanup
   - **Cost**: $0 for embeddings! 🎉

## 📚 Quick Reference

### Vector Database Schema (Weaviate)

```json
{
  "vector": [768 floats],  // nomic-embed-text embeddings
  "text": "document content",
  "source": "https://js.langchain.com/docs/...",
  "title": "Page Title | 🦜️🔗 LangChain.js"
}
```

### Record Manager Schema (PostgreSQL)

```json
{
  "key": "https://js.langchain.com/docs/...",
  "content_hash": "a3f5c9b2e8d1...",
  "updated_at": "2024-10-30 10:30:00",
  "namespace": "weaviate/LangChain_General_Guides_..."
}
```

### Index Function Parameters

```python
index(
    docs,                    # Documents to process
    record_manager,          # Deduplication tracker
    vectorstore,             # Storage destination
    cleanup="full",          # Auto-delete removed documents
    source_id_key="source",  # Group by URL
    force_update=False       # Skip if unchanged
)
```

## 🚀 Running the System

### Start Docker Containers

```bash
# Start all services (PostgreSQL, Weaviate, Ollama)
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f weaviate
```

### Run Ingestion

```bash
# Set environment (already configured in .env)
# RECORD_MANAGER_DB_URL=postgresql://postgres:zkdtn1234@localhost:5432/chat_langchain

# Run ingestion (make sure Ollama is running and has pulled nomic-embed-text)
PYTHONPATH=. poetry run python backend/ingest.py

# Output:
# Fetching pages: 100%|████████| 32/32 [00:08<00:00,  3.60it/s]
# INFO: Indexing stats: {'num_added': 120, 'num_updated': 0, 'num_skipped': 0, 'num_deleted': 0}
# INFO: General Guides and Tutorials now has this many vectors: 120
```

### Query Documents

```bash
# Test retrieval
PYTHONPATH=. poetry run python -c "
from langchain_core.runnables import RunnableConfig
from backend.retrieval import make_retriever

config = RunnableConfig(configurable={'search_kwargs': {'k': 3}})
with make_retriever(config) as retriever:
    docs = retriever.invoke('What is LangChain?')
    print(f'Found {len(docs)} documents')
    print(f'Top result: {docs[0].metadata[\"title\"][:60]}')
"
```

---

**Built with:** LangChain, Weaviate, PostgreSQL, Ollama, nomic-embed-text

**Architecture:** Local Docker setup with Ollama-powered embeddings

**Last Updated:** October 30, 2024
