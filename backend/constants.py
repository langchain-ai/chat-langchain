import os

# Weaviate index names
WEAVIATE_DOCS_INDEX_NAME = "LangChain_Combined_Docs_nomic_embed_text"
WEAVIATE_GENERAL_GUIDES_AND_TUTORIALS_INDEX_NAME = (
    "LangChain_General_Guides_And_Tutorials_nomic_embed_text"
)

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

OLLAMA_BASE_EMBEDDING_DOCS_URL = os.getenv(
    "OLLAMA_BASE_EMBEDDING_DOCS_URL", OLLAMA_BASE_URL
)
