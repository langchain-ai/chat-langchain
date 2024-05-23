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

1. Make sure you have all 3 secret files (`.env.secret`, `.env.share`, `dsmain_ssh_ec3`) in folder `secrets`
2. Launch app: `docker compose -f ./docker-compose-local.yml up -d --build`
3. The app is now available at `http://localhost:3000/`
4. To run tests, open a terminal:

   - go into running backend container: `docker exec -ti $(docker ps -qf "name=chat-langchain-backend") /bin/bash`
   - run tests: `python -m pytest tests`
   - run evaluation script whose options/args are:

     ```
     root@e6123c0f9b6b:~# python _scripts/evaluate_doc_retrieval.py --help
     usage: evaluate_doc_retrieval.py [-h] [--use-model-llm] eval_path

     positional arguments:
     eval_path        CSV file path that contains evaluation use cases

     options:
      -h, --help       show this help message and exit
      --use-model-llm  Option which, when specified, tells the evaluation to use model_llm (i.e. use model_openai_functions when this option is not specified)
     ```

     You can see an example of the script's expected input in `./_scripts/evaluate_doc_retrieval.csv`.
     You can then use `pandas.read_csv(<path>)` to load the generated evaluation report (whose path is reported on the last line of the script).

     ```
      root@e6123c0f9b6b:~# python _scripts/evaluate_doc_retrieval.py --use-model-llm ./_scripts/evaluate_doc_retrieval.csv
      INFO:root:Evaluating croptalk's document retrieval capacity, using config: Namespace(use_model_llm=True, eval_path='./_scripts/evaluate_doc_retrieval.csv')

      INFO:root:Number of use cases to evaluate: 2
      INFO:root:Creating output_df
      INFO:root:Loading model

      (...)

      INFO:root:Evaluation report/dataframe saved here: ./_scripts/evaluate_doc_retrieval__model_llm__2024-02-23T22:07:10.813830.csv
     ```

## Running in the interactive mode (notebooks enabled)

1. Launch the container
   `docker-compose -f docker-compose-local.yml up -d --build`
2. Attach to docker through **VSCode Remote Explorer**
3. Open an .ipynb and select a Python kernel. Install python and jupyter if needed (they are not installed in the container by default)

## Testing the performance

With the running docker, execute the script (Modify dataset name if needed):
`docker exec -it chat-langchain-backend-1 python _scripts/evaluate_overall_performance.py`

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
