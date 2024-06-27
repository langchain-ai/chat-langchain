# 🦜️🔗 Chat LangChain

This repo is an implementation of a chatbot specifically focused on question answering over the [LangChain documentation](https://python.langchain.com/).
Built with [LangChain](https://github.com/langchain-ai/langchain/), [FastAPI](https://fastapi.tiangolo.com/), and [Next.js](https://nextjs.org).

Deployed version: [chat.langchain.com](https://chat.langchain.com)

> Looking for the JS version? Click [here](https://github.com/langchain-ai/chat-langchainjs).

The app leverages LangChain and LangGraph's streaming support and async API to update the page in real time for multiple users.

Question-Answering has the following steps:

1. Given the chat history and new user input, determine what a standalone question would be using an LLM.
2. Given that standalone question, look up relevant documents from the vectorstore.
3. Pass the standalone question and relevant documents to the model to generate and stream the final answer.
4. Generate a trace URL for the current chat session, as well as the endpoint to collect feedback.

## Documentation

Looking to use or modify this Use Case Accelerant for your own needs? We've added a few docs to aid with this:

- **[Concepts](./CONCEPTS.md)**: A conceptual overview of the different components of Chat LangChain. Goes over features like ingestion, vector stores, query analysis, etc.
- **[Modify](./MODIFY.md)**: A guide on how to modify Chat LangChain for your own needs. Covers the frontend, backend and everything in between.
- **[LangSmith](./LANGSMITH.md)**: A guide on adding robustness to your application using LangSmith. Covers observability, evaluations, and feedback.
- **[Production](./PRODUCTION.md)**: Documentation on preparing your application for production usage. Explains different security considerations, and more.
- **[Deployment](./DEPLOYMENT.md)**: How to deploy your application to production. Covers setting up production databases, deploying the frontend, and more.