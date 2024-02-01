# CropTalk

Create testing dataset
`python croptalk/create_test_dataset.py`
Run on the dataset
`python evaluate_openai_functions.py`
Check Langsmith datasets for the results.

# 🔗 Chat LangChain

This repo is an implementation of a locally hosted chatbot specifically focused on question answering over the [LangChain documentation](https://langchain.readthedocs.io/en/latest/).
Built with [LangChain](https://github.com/hwchase17/langchain/), [FastAPI](https://fastapi.tiangolo.com/), and [Next.js](https://nextjs.org).

Deployed version: [chat.langchain.com](https://chat.langchain.com)

The app leverages LangChain's streaming support and async API to update the page in real time for multiple users.

## ✅ Running locally
0. Update lock `poetry lock`
1. Install backend dependencies: `poetry install`.
1. Make sure to enter your environment variables to configure the application:
```
export OPENAI_API_KEY=
export WEAVIATE_URL=
export WEAVIATE_API_KEY=
export RECORD_MANAGER_DB_URL=

# for tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_API_KEY=
export LANGCHAIN_PROJECT=
```
1. Run `python ingest.py` to ingest LangChain docs data into the Weaviate vectorstore (only needs to be done once).
   1. You can use other [Document Loaders](https://langchain.readthedocs.io/en/latest/modules/document_loaders.html) to load your own data into the vectorstore.
1. Start the Python backend with `poetry run make start`.
1. Install frontend dependencies by running `cd chat-langchain`, then `yarn`.
1. Run the frontend with `yarn dev` for frontend.
1. Open [localhost:3000](http://localhost:3000) in your browser.

### Running locally... in docker

1. Modify docker-compose.yml:
   - Un-comment line where `.` is mounted in `/app` within container (so that local code changes are reflected in running container)
   - Comment line where `entrypoint` is currently defined, and un-comment the line where it is defined with `--reload` (so that backend container automatically detect code changes)
   - Comment line where `GRAPH_URL_HOST` is currently set, and un-comment the line where it is set to `http://localhost` instead
   - Comment line where `NEXT_PUBLIC_API_BASE_URL` is currently set, and un-comment the line where it is set to `http://localhost` instead
2. Launch app: `docker compose up -d --build`
3. The app is now available at `http://localhost:3000/`
4. To run tests, open a terminal:
   - go into running backend container: `docker exec -ti <container_id> /bin/bash`
   - run tests: `python -m pytest tests`


## 📚 Technical description

There are two components: ingestion and question-answering.

Ingestion has the following steps:

1. Pull html from documentation site as well as the Github Codebase
2. Load html with LangChain's [RecursiveURLLoader](https://python.langchain.com/docs/integrations/document_loaders/recursive_url_loader) and [SitemapLoader](https://python.langchain.com/docs/integrations/document_loaders/sitemap)
3. Split documents with LangChain's [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain.text_splitter.RecursiveCharacterTextSplitter.html)
4. Create a vectorstore of embeddings, using LangChain's [Weaviate vectorstore wrapper](https://python.langchain.com/docs/integrations/vectorstores/weaviate) (with OpenAI's embeddings).

Question-Answering has the following steps:

1. Given the chat history and new user input, determine what a standalone question would be using GPT-3.5.
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to the model to generate and stream the final answer.
4. Generate a trace URL for the current chat session, as well as the endpoint to collect feedback.

## 🚀 Deployment

Deploy the frontend Next.js app as a serverless Edge function on Vercel [by clicking here]().
You'll need to populate the `NEXT_PUBLIC_API_BASE_URL` environment variable with the base URL you've deployed the backend under (no trailing slash!).
