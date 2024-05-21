# Modifying

The Chat LangChain repo was built to serve two use cases.
The first being question answering over the LangChain documentation.
The second is to offer a production ready chat bot which you can easily customize for your specific use case.
In this doc we'll go over each step you need to take to customize the repo for your need.

## Vector Store

One of the simplest ways to modify Chat LangChain and get a feel for the codebase is to modify the vector store.
All of the operations in Chat LangChain are largely based around the vector store:

- ingestion
- retrieval
- context
- etc

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

## Record Manager

Continuing with the database, we also employ a record manager for ingesting docs.
Currently, we use a `SQLRecordManager`, however you may also swap that out in favor of a `MongoDocumentManager`:

```python
from langchain_community.indexes import MongoDocumentManager

record_manager = MongoDocumentManager(
    namespace="kittens",
    mongodb_url="mongodb://langchain:langchain@localhost:6022/",
    db_name="test_db",
    collection_name="test_collection",
)
record_manager.create_schema()
```

For more conceptual information on Record Managers with LangChain, see the [concepts](./CONCEPTS.md) doc.

## LLM

The LLM is used inside the `/chat` endpoint for generating the final answer, and performing query analysis on followup questions.

> Want to learn more about query analysis? See our comprehensive set of use case docs [here](https://python.langchain.com/docs/use_cases/query_analysis/).

Without any modification, we offer a few LLM providers out of the box:

- `gpt-3.5-turbo-0125` by OpenAI
- `claude-3-haiku-20240307` by Anthropic
- `mixtral-8x7b` by Fireworks
- `gemini-pro` by Google
- `command` by Cohere

These are all located at the bottom of the [`./backend/chain.py`](./backend/chain.py) file. You have a few options for modifying this:

- Replace all options with a single provider
- Add more providers

First, I'll demonstrate how to replace all options with a single provider, as it's the simplest:

1. Find the LLM variable declaration at the bottom of the file, it looks something like this:

```python
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    streaming=True,
    temperature=0,
).configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    default_key="openai_gpt_3_5_turbo",
    anthropic_claude_3_haiku=ChatAnthropic(
        model="claude-3-haiku-20240307",
        max_tokens=16384,
        temperature=0,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "not_provided"),
    ),
    ...
)
```

You should then remove it, and replace with your LLM class of choice, imported from LangChain. Remember to keep the variable name the same so nothing else in the endpoint breaks:

```python
llm = ChatYourLLM(
    model="model-name",
    streaming=True,
    temperature=0,
).configurable_alternatives(
    # This gives this field an id
    ConfigurableField(id="llm")
)
```

Adding alternatives is also quite simple. Just add another class declaration inside the `configurable_alternatives` method. Here's an example:

```python
.configurable_alternatives(
    local_ollama=ChatCohere(
        model="llama2",
        temperature=0,
    ),
)
```

Next, scroll up to find the `response_synthesizer` variable, and add an entry for `local_ollama` like so:

```python
response_synthesizer = (
    default_response_synthesizer.configurable_alternatives(
        ConfigurableField("llm"),
        default_key="openai_gpt_3_5_turbo",
        anthropic_claude_3_haiku=default_response_synthesizer,
        ...
        local_ollama=default_response_synthesizer,
    )
    | StrOutputParser()
).with_config(run_name="GenerateResponse")
```

That't it!

## Embeddings

Chat LangChain uses embeddings inside the ingestion script when storing documents in the vector store.
Without modification, it defaults to use [OpenAI's embeddings model](https://python.langchain.com/docs/integrations/text_embedding/openai).

Changing this to the vector store of your choice is simple. First, find the `get_embeddings_model` function inside the [`./backend/ingest.py`](./backend/ingest.py) file. It looks something like this:

```python
def get_embeddings_model() -> Embeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=200)
```

Then, simply swap out the `OpenAIEmbeddings` class for the model of your choice!

Here's an example of what that would look like if you wanted to use Mistral's embeddings model:

```python
from langchain_mistralai import MistralAIEmbeddings

def get_embeddings_model() -> Embeddings:
    return MistralAIEmbeddings(mistral_api_key="your-api-key")
```

## Prompts

### Answer Generation Prompt

The prompt used for answer generation is one of the most important parts of this RAG pipeline. Without a good prompt, the LLM will be unable (or severely limited) to generate good answers.

The prompt is defined in the `RESPONSE_TEMPLATE` variable.

You should modify the parts of this which are LangChain specific to instead fit your needs. If possible, and if your use case does not differ too much from the Chat LangChain use case, you should do your best to keep the same structure as the original prompt. Although there are likely some improvements to be made, we've refined this prompt over many months and lots of user feedback to what we believe to be a well formed prompt.

### Question Rephrasing Prompt

Finally, you can (but not necessary required) modify the `REPHRASE_TEMPLATE` variable to contain more domain specific content about, for example, the types of followup questions you expect to receive. Having a good rephrasing prompt will help the LLM to better understand the user's question and generate a better prompt which will have compounding effects downstream.

## Retrieval

### Ingestion Script

At a high level, the only LangChain specific part of the ingestion script are the three webpages which is scrapes for documents to add to the vector store. These links are:
- LangSmith Documentation
- LangChain API references
- LangChain Documentation

If all you would like to update is update which website(s) to scrape and ingest, you only need to modify/remove these functions:

- `load_langchain_docs`
- `load_langsmith_docs`
- `load_api_docs`

Other than this, the core functionality of the retrieval system is not LangChain specific.

### Retrieval Methods

You can however, easily add or remove parts to increase/fit your needs better.
Some ideas of what can be done:

- Re-ranking document results
- Parent document retrieval (also would require modifications to the ingestion script)
- Document verification via LLM

## Frontend

The frontend doesn't have much LangChain specific code that would need modification.
The main parts are the LangChain UI branding and question suggestions.

To modify the main LangChain branding visit the [`ChatWindow.tsx`](frontend/app/components/ChatWindow.tsx) file to modify/remove.
Next, to update the question suggestions, visit the [`EmptyState.tsx`](frontend/app/components/EmptyState.tsx) file.
Finally, update the "View Source" button on the bottom of the page by going back to the [`ChatWindow.tsx`](frontend/app/components/ChatWindow.tsx) file.