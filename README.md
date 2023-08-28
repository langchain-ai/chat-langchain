# 🦜️🔗 Chat LangChain

This repo is an implementation of a locally hosted chatbot specifically focused on question answering over the [LangChain documentation](https://langchain.readthedocs.io/en/latest/).
Built with [LangChain](https://github.com/hwchase17/langchain/) and [FastAPI](https://fastapi.tiangolo.com/).

The app leverages LangChain's streaming support and async API to update the page in real time for multiple users.

## ✅ Running locally
1. Install dependencies: `pip install -r requirements.txt`
1. Run `python ingest.py` to ingest LangChain docs data into the Weaviate vectorstore (only needs to be done once).
   1. You can use other [Document Loaders](https://langchain.readthedocs.io/en/latest/modules/document_loaders.html) to load your own data into the vectorstore.
1. Run the app: `make start` for backend and `npm run dev` for frontend (cd into chat-langchain first)
   1. Make sure to enter your environment variables to configure the application: 
   ```
   export OPENAI_API_KEY=
   export WEAVIATE_URL=
   export WEAVIATE_API_KEY=
   
   # for tracing
   export LANGCHAIN_TRACING_V2=true
   export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   export LANGCHAIN_API_KEY=
   export LANGCHAIN_PROJECT=
   ```
1. Open [localhost:3000](http://localhost:3000) in your browser.

## 🚀 Important Links

Deployed version: [chat.langchain.com](https://chat.langchain.com)


## 📚 Technical description

There are two components: ingestion and question-answering.

Ingestion has the following steps:

1. Pull html from documentation site as well as the Github Codebase
2. Load html with LangChain's [RecursiveURLLoader Loader](https://python.langchain.com/docs/integrations/document_loaders/recursive_url_loader)
2. Transform html to text with [Html2TextTransformer](https://python.langchain.com/docs/integrations/document_transformers/html2text)
3. Split documents with LangChain's [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)
4. Create a vectorstore of embeddings, using LangChain's [Weaviate vectorstore wrapper](https://python.langchain.com/docs/integrations/vectorstores/weaviate) (with OpenAI's embeddings).

Question-Answering has the following steps, all handled by [OpenAIFunctionsAgent](https://python.langchain.com/docs/modules/agents/agent_types/openai_functions_agent):

1. Given the chat history and new user input, determine what a standalone question would be (using GPT-3.5).
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to GPT-4 to generate and stream the final answer.
4. Generate a trace URL for the current chat session, as well as the endpoint to collect feedback. 

## Deprecated Links
Hugging Face Space (to be updated soon): [huggingface.co/spaces/hwchase17/chat-langchain](https://huggingface.co/spaces/hwchase17/chat-langchain)

Blog Posts: 
* [Initial Launch](https://blog.langchain.dev/langchain-chat/)
* [Streaming Support](https://blog.langchain.dev/streaming-support-in-langchain/)