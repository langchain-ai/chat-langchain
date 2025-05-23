# 🦜️🔗 Chat LangChain - Custom Support Bot

This project is a customer support chatbot built using LangChain, LangServe, Milvus, and Ollama, with a Streamlit frontend. It is based on the original `chat-langchain` repository but adapted for PDF document ingestion and local model support.

## Features

*   Ingests PDF documents from a local folder (`knowledge_docs`).
*   Extracts text from PDFs for question answering.
*   **Placeholder logic for extracting embedded source URLs from PDFs for citations (requires user modification).**
*   Uses Milvus as the vector store for efficient document retrieval.
*   Supports OpenAI and Ollama for both embeddings and language models, configurable via environment variables.
*   Provides an interactive Streamlit chat interface for user interaction.
*   Includes source citations in responses (dependent on successful URL extraction).

## Setup & Installation

### Prerequisites

*   Python 3.9+
*   Poetry for dependency management
*   A running Milvus instance (version 2.3.x or compatible).
*   Ollama server running (optional, if you intend to use Ollama models).
    *   Ensure you have pulled the necessary models (e.g., `ollama pull llama2`, `ollama pull nomic-embed-text`).

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_name>     # Replace <repository_name> with the cloned directory name
    ```

2.  **Install dependencies:**
    This project uses Poetry to manage dependencies.
    ```bash
    poetry install --no-root
    ```
    This command installs all necessary Python packages defined in `pyproject.toml`.

3.  **Set up Environment Variables:**
    Copy the example environment file to a new `.env` file:
    ```bash
    cp .env.example .env
    ```
    Then, edit the `.env` file and provide the necessary values. Below are the key variables:

    *   `LANGCHAIN_TRACING_V2`: (Optional) Set to `"true"` to enable LangSmith tracing.
    *   `LANGCHAIN_ENDPOINT`: (Optional) LangSmith API endpoint, if tracing.
    *   `LANGCHAIN_API_KEY`: (Optional) Your LangSmith API key, if tracing.
    *   `LANGCHAIN_PROJECT`: (Optional) Your LangSmith project name, if tracing.

    *   `EMBEDDING_PROVIDER`: Specifies the embedding model provider.
        *   Set to `"openai"` to use OpenAI embeddings.
        *   Set to `"ollama"` to use a local Ollama embedding model.
        *   *Default: "openai"*
    *   `LLM_PROVIDER`: Specifies the language model provider for the chat.
        *   Set to `"openai"` to use an OpenAI model.
        *   Set to `"ollama"` to use a local Ollama model.
        *   *Default: "openai"*

    *   `OPENAI_API_KEY`: **Required if `EMBEDDING_PROVIDER` or `LLM_PROVIDER` is "openai".** Your API key for OpenAI.

    *   `OLLAMA_BASE_URL`: **Required if `EMBEDDING_PROVIDER` or `LLM_PROVIDER` is "ollama".** The base URL for your Ollama server.
        *   *Default: "http://localhost:11434"*
    *   `OLLAMA_EMBEDDING_MODEL`: The name of the embedding model to use with Ollama (e.g., "nomic-embed-text").
        *   *Default: "nomic-embed-text"*
    *   `OLLAMA_LLM_MODEL`: The name of the language model to use with Ollama (e.g., "llama2").
        *   *Default: "llama2"*

    *   `MILVUS_HOST`: The hostname or IP address of your Milvus instance.
        *   *Default: "localhost"*
    *   `MILVUS_PORT`: The port number for your Milvus instance.
        *   *Default: "19530"*
    *   `MILVUS_COLLECTION_NAME`: The name of the collection to be used in Milvus for storing document embeddings.
        *   *Default: "knowledge_base"*

    *   `RECORD_MANAGER_DB_URL`: The database URL for the record manager, which keeps track of ingested documents. SQLite is used by default.
        *   *Example: "sqlite:///./chat_langchain_ingestion.db"*

## Data Preparation (IMPORTANT)

1.  **Place PDF Files:**
    Put all your PDF documents that you want the chatbot to use into the `knowledge_docs` folder located in the root of this repository. If this folder does not exist, please create it.

2.  **Implement Source URL Extraction (User Action Required):**
    The current system for extracting source URLs from PDFs (to provide accurate citations) uses a **placeholder**. You **must** modify the PDF ingestion logic to correctly extract the source URL for each document.

    *   **File to Modify:** `backend/ingest.py`
    *   **Function to Modify:** `load_knowledge_base_pdfs`
    *   **Area to Modify:** Look for the variable `extracted_url`. Currently, it is hardcoded:
        ```python
        extracted_url = "placeholder_url_needs_implementation"
        ```
    *   **Your Task:** Implement logic to find and assign the correct source URL for each PDF. This could involve:
        *   Parsing the PDF content for a specific line or pattern (e.g., a line starting with "Source: http://..." or "Canonical URL: https://...").
        *   Using a sidecar metadata file for each PDF.
        *   Deriving the URL from the PDF's filename if it follows a consistent pattern.
        *   Querying an external system based on PDF content or filename.

    **Example of what you might look for in a PDF's text content:**
    ```
    Source: https://my-company.com/original-document.pdf
    ```
    Or it might be embedded in the PDF's metadata if the generating system includes it. The `PyPDFLoader` loads document pages; you might need to inspect `page.page_content` or `page.metadata` for clues.

    **Failure to implement this step will result in all sources being cited with the placeholder URL.**

## Running the Application

Ensure your Milvus instance (and Ollama server, if using it) is running before starting the application.

1.  **Step 1: Run Ingestion (One-time per data update)**
    This script processes the PDFs in `knowledge_docs`, creates embeddings, and stores them in Milvus. Run it once initially, and then again whenever you add, remove, or change PDF files.
    ```bash
    poetry run python backend/ingest.py
    ```

2.  **Step 2: Run Backend Server (LangServe)**
    This starts the FastAPI server that serves your chat graph using LangServe.
    ```bash
    poetry run uvicorn backend.server:app --reload --host 0.0.0.0 --port 8000
    ```
    The backend will be accessible at `http://localhost:8000`.

3.  **Step 3: Run Streamlit Frontend**
    In a new terminal, start the Streamlit application.
    ```bash
    poetry run streamlit run streamlit_app.py
    ```
    Streamlit will typically open in your web browser automatically, or provide a URL (usually `http://localhost:8501`).

## Usage

1.  Open the Streamlit application URL in your web browser (e.g., `http://localhost:8501`).
2.  In the sidebar, you can:
    *   Verify or change the LangServe URL (defaults to `http://localhost:8000/chat_langchain/`).
    *   Select the LLM model you wish to use from the dropdown menu. The available models depend on your backend configuration and environment variables (e.g., OpenAI keys, Ollama setup).
3.  Type your question related to the content of your ingested PDF documents into the chat input box at the bottom of the page and press Enter.
4.  The chatbot will process your question, retrieve relevant information from the documents in Milvus, generate an answer, and display it along with source citations.

---

This README provides a comprehensive guide to setting up and running the custom chatbot. Remember to implement the PDF source URL extraction logic for accurate citations.

## Project Structure and Key Files

Here's a brief overview of the key files and directories in this project:

*   **`streamlit_app.py`**: The main Streamlit application file that provides the user interface for the chatbot. You run this file to start the chat application.
*   **`backend/`**: This directory contains all the Python backend logic.
    *   **`backend/server.py`**: The FastAPI server that exposes the LangServe chat graph as an API. This is what the Streamlit frontend communicates with.
    *   **`backend/graph.py`**: Defines the core chat logic using LangGraph. It constructs the conversational chain, including document retrieval, history management, LLM calls, and response synthesis.
    *   **`backend/ingest.py`**: Handles the data ingestion pipeline. It reads PDFs from the `knowledge_docs` folder, extracts text (and is designed for you to add logic to extract source URLs), generates embeddings, and stores them in the Milvus vector database.
    *   **`backend/constants.py`**: Originally for Weaviate constants; now mostly cleaned up. Can be used for any backend-specific global constants if needed.
*   **`knowledge_docs/`**: This is the directory where you must place your PDF documents for ingestion. The `ingest.py` script processes files from here.
*   **`.env.example`**: A template file for environment variables. You should copy this to a `.env` file and fill in your specific configurations (API keys, Milvus/Ollama URLs, etc.).
*   **`.env`**: (User-created) This file stores your actual environment variable configurations. It's listed in `.gitignore` and should not be committed to version control.
*   **`pyproject.toml`**: Defines project dependencies and metadata for Poetry.
*   **`poetry.lock`**: The lock file generated by Poetry, ensuring consistent dependency versions.