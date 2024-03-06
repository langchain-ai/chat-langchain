# Concepts

In this doc we'll go over the different concepts that are implemented in Chat LangChain.
By the end, you'll have a conceptual understanding of how Chat LangChain works, and it's different architectural components.
We'll start with the vector store, the basis of the entire system.

## Vector Store

Vector stores, fundamentally, are specialized databases designed to efficiently store and manage vectors, which are high-dimensional arrays of numbers. These vectors are not arbitrary; they are the product of sophisticated text embedding models, such as those provided by [OpenAI's `text-embedding`](https://platform.openai.com/docs/guides/embeddings) API.

In the context of our application, vector stores play a pivotal role in enhancing the capabilities of our language model. Here's a deeper dive into the process:

1. **Vector Generation**: Whenever new content related to LangChain is introduced or existing content is updated, we use text embedding models to convert this textual information into vectors. Each vector acts as a unique fingerprint of its corresponding text, encapsulating its meaning in a high-dimensional space.

2. **Similarity Searches**: The core utility of storing these vectors comes into play when we need to find information relevant to a user's query. By converting the user's question into a vector using the same embedding model, we can perform a similarity search across our vector store. This search identifies vectors (and thus, documents) whose meanings are closest to the query, based on the distance between vectors in the embedding space.

3. **Context Retrieval and Enhancement**: The documents retrieved through similarity searches are relevant pieces of information that aid the language model in generating relevant answers. By providing this context, we enable the language model to generate responses that are not only accurate but also informed by the most relevant and up-to-date information available in our database.

## Indexing

Indexing your documents is a vital part of any production RAG application. In short, indexing allows for your documents to be stored, and searchable to prevent duplicate documents from being stored. This is important for a few reasons:

1. **Duplicate Results**: Say you update your vector store without using an Indexing API. Now you may have two identical documents in your store. Then, when you perform a semantic search, instead of getting K number of different results, you'll get duplicates as the semantic search only returns documents which are semantically close to the query.

2. **Performance**: Indexing your documents allows for faster ingestion. With indexing you don't have to generate embeddings for every document on ingestion, and instead only need to generate embeddings for new documents.

In order to help with indexing we use the LangChain indexing API. This API contains all the features required for robust indexing in your application. Indexing is done in two main steps:

1. **Ingestion**: Ingestion is where you pull in all the documents you want to add to your vector store. This could be all of the documents available to you, or just a couple new documents.
2. **Hashing**: Once the ingestion API is passed your documents, it creates a unique hash for each, containing some metadata like the date it was ingested. This allows for the indexing API to only ingest new documents, and not duplicate documents. These hashes are stored in what we call the "Record Manager".
3. **Insertion**: Finally, once the documents are hashed, and confirmed to not already exist through the Record Manager, they are inserted into the vector store.

The indexing API also uses a Record Manager to store the records of previously indexed documents in between ingestion. This manager stores the hashed values of the documents, and the time they were ingested. This allows for the indexing API to only ingest new documents, and not duplicate documents.

## Query Analysis

Finally, we perform query analysis on followup chat conversations. It is important to note that we only do this for followups, and not initial questions. Let's break down the reasoning here:

Users are not always the best prompters, and can very easily miss some context or phrase their question poorly. We can be confident that the LLM will not make this mistake.
Additionally, given a chat history (which is always passed in context to a model) you may not need to include certain parts of the question, or the reverse, where you do need to clarify additional information.

Doing all this helps make better formed questions for the model, without having to rely on the user to do so.

Lastly, we don't perform this on the initial question for two main reasons:

1. **Speed**: Although models are getting faster and faster, they still take longer than we'd like to return a response. This is even more important for the first question, as the chat bot hasn't proved its usefulness to the user yet, and you don't want to lose them due to speed before they've even started.
2. **Context**: Without a chat history, the model is lacking some important context around the users question.

Most users won't format their queries perfectly for LLMs, and that's okay!
To account for this, we have an extra step before final generation which takes the users query and rephrase it to be more suitable for the LLM.

The prompt is quite simple:
```typescript
const REPHRASE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:`;
```

In doing this, the language model is able to take the users question, and the full chat history which contains other questions, answers and context, and generate a more well formed response. Now using this rephrased question, we can perform a similarity search on the vector store using this question, and often times get back better results as the question is semantically more similar to the previous questions/answers (content the database).