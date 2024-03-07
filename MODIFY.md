# Modifying

The Chat LangChain repo was built to serve two use cases. The first being question answering over the LangChain documentation. The second is to offer a production ready chat bot which you can easily customize for your specific use case. In this doc we'll go over each step you need to take to customize the repo for your need.

## Vector Store

One of the simplest ways to modify Chat LangChain and get a feel for the codebase is to modify the vector store. All of the operations in Chat LangChain are largely based around the vector store: ingestion, retrieval, context, etc.

There are two places the vector store is used:
- **Ingestion**: The vector store is used to store the embeddings of every document used as context. Located in [`./backend/ingest.py`](./backend/ingest.py) you can easily modify the provider to use a different vector store.
- **Retrieval**: The vector store is used to retrieve documents based on a user's query. Located at [`./backend/chain.py`](./backend/chain.py) you can easily modify the provider to use a different vector store.

### Steps

For backend ingestion, locate the `ingest_docs` function. You'll want to modify where `client` and `vectorstore` are instantiated. Here's an example of the Weaviate instantiation:

```python
client = weaviate.Client(
    url=WEAVIATE_URL,
    auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
)
vectorstore = Weaviate(
    client=client,
    index_name=WEAVIATE_DOCS_INDEX_NAME,
    text_key="text",
    embedding=embedding,
    by_text=False,
    attributes=["source", "title"],
)
```

To make transitioning as easy as possible, all you should do is:

1. Delete the weaviate client instantiation.
2. Replace the vector store instantiation with the new provider's instantiation. Remember to keep the variable name (`vectorstore`) the same. Since all LangChain vector stores are built on top of the same API, no other modifications should be necessary.

Finally, perform these same steps inside the [`./backend/chain.py`](./backend/chain.py) (inside the `get_retriever` function) file, and you're done!

## Frontend

The frontend doesn't have much LangChain specific code that would need modification. The main parts are the LangChain UI branding and question suggestions.

To modify the main LangChain branding visit the [`ChatWindow.tsx`](frontend/app/components/ChatWindow.tsx) file to modify/remove. Next, to update the question suggestions, visit the [`EmptyState.tsx`](frontend/app/components/EmptyState.tsx) file. Finally, update the "View Source" button on the bottom of the page by going back to the [`ChatWindow.tsx`](frontend/app/components/ChatWindow.tsx) file.

## Backend (API)

The backend contains two main parts, both of which live inside the [`./backend`](/backend/) directory.
The first is the ingestion script which lives inside of [`./backend/ingest.py`](./backend/ingest.py).
The second are the API endpoints which are defined inside [`./backend/main.py`](backend/main.py), and most of the logic living inside [`./backend/chain.py`](./backend/chain.py).

### Ingestion Script

We'll start by modifying the ingestion script.

At a high level, the only LangChain specific part of the ingestion script are the three webpages which is scrapes for documents to add to the vector store. These links are:
- LangSmith Documentation
- LangChain API references
- LangChain Documentation

If all you would like to update is update which website(s) to scrape and ingest, you only need to modify/remove these functions:

- `load_langchain_docs`
- `load_langsmith_docs`
- `load_api_docs`

If you want to ingest another way, consult the [document loader](https://python.langchain.com/docs/modules/data_connection/document_loaders/) section of the LangChain docs.
Using any LangChain document loader, you'll be able to easily and efficiently fetch & ingest from a large variety of sources. Additionally, the LangChain. document loader API will always return documents in the same format ([`Document`](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html)) so you do not need to modify the format before adding to your indexing API or vector store.

### API Endpoints

Once you've updated your ingestion script to populate your vector store, you'll need to update the different API endpoints to reflect your new data. 

The first (and main endpoint) is `/chat`. This is where the main API code lives.

- Answer generation prompt
- Question rephrasing (query analysis) prompt
- Document retrieval

#### Document Retrieval

The document retrieval code is the only part of this API which is not LangChain documentation specific. You can however, easily add or remove parts to increase/fit your needs better. Some ideas of what can be done:

- Re-ranking document results
- Parent document retrieval (also would require modifications to the ingestion script)
- Document verification via LLM

#### Answer Generation Prompt

The prompt used for answer generation is one of the most important parts of this RAG pipeline. Without a good prompt, the LLM will be unable (or severely limited) to generate good answers.

The prompt is defined in the `RESPONSE_TEMPLATE` variable.

You should modify the parts of this which are LangChain specific to instead fit your needs. If possible, and if your use case does not differ too much from the Chat LangChain use case, you should do your best to keep the same structure as the original prompt. Although there are likely some improvements to be made, we've refined this prompt over many months and lots of user feedback to what we believe to be a well formed prompt.

#### Question Rephrasing Prompt

Finally, you can (but not necessary required) modify the `REPHRASE_TEMPLATE` variable to contain more domain specific content about, for example, the types of followup questions you expect to receive. Having a good rephrasing prompt will help the LLM to better understand the user's question and generate a better prompt which will have compounding effects downstream.


# ------------New structure proposal------------

# Modifying

## Vector Store

## Record Manager

## Embedding Model

## LLM

## Prompts

## Retrieval

## Frontend
