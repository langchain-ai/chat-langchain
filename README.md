# 🦜️🔗 ChatLangChain

This repo is an implementation of a locally hosted chatbot specifically focused on question answering over the [LangChain documentation](https://langchain.readthedocs.io/en/latest/).
Built with [LangChain](https://github.com/hwchase17/langchain/) and [FastAPI](https://fastapi.tiangolo.com/).

The app leverages LangChain's streaming support and async API to update the page in real time for multiple users.

## ✅ To run:
1. Install dependencies: `pip install -r requirements.txt`
1. Run `./ingest.sh` to ingest LangChain docs data into the vectorstore (only needs to be done once).
1. To see an example of how to ingest other docs, see `ingest_state_of_union.py`.
1. Run the app: `make start`
1. To enable tracing, make sure `langchain-server` is running locally and pass `tracing=True` to `get_chain` in `main.py`.
1. Open [localhost:9000](http://localhost:9000) in your browser.

<a href="https://www.loom.com/share/a64b1def314a4884ab0526bf77d9fa65">
    <p><strong>Chat Your Data with `state_of_the_union.txt`</strong></p>
    <img style="max-width:800px;" src="https://cdn.loom.com/sessions/thumbnails/a64b1def314a4884ab0526bf77d9fa65-1676325415887-with-play.gif">
  </a>

## 🚀 Important Links

Deployed version (to be updated soon): [chat.langchain.dev](https://chat.langchain.dev)

Hugging Face Space (to be updated soon): [huggingface.co/spaces/hwchase17/chat-langchain](https://huggingface.co/spaces/hwchase17/chat-langchain)

Blog Posts: 
* [blog.langchain.dev/langchain-chat/](https://blog.langchain.dev/langchain-chat/)

## 📚 Technical description

There are two components: ingestion and question-answering.

Ingestion has the following steps:

1. Pull html from documentation site
2. Parse html with BeautifulSoup
3. Split documents with LangChain's [TextSplitter](https://langchain.readthedocs.io/en/latest/modules/utils/combine_docs_examples/textsplitter.html)
4. Create a vectorstore of embeddings, using LangChain's [vectorstore wrapper](https://langchain.readthedocs.io/en/latest/modules/utils/combine_docs_examples/vectorstores.html) (with OpenAI's embeddings and Weaviate's vectorstore).

Question-Answering has the following steps, all handled by [ChatVectorDBChain](https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs_examples/chat_vector_db.html):

1. Given the chat history and new user input, determine what a standalone question would be (using GPT-3).
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to GPT-3 to generate a final answer.

## 🧠 How to Extend to your documentation?

See `ingest_state_of_union.py` for an example of how to ingest your own docs.
