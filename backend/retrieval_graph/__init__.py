"""Retrieval Graph Module

This module provides an intelligent conversational retrieval graph system for
handling user queries about LangChain and related topics.

The main components of this system include:

1. A state management system for handling conversation context and research steps.
2. An analysis and routing mechanism to classify user queries and determine the appropriate response path.
3. A research planner that breaks down complex queries into manageable steps.
4. A researcher agent that generates queries and fetches relevant information based on research steps.
5. A response generator that formulates answers using retrieved documents and conversation history.

The graph is configured using customizable parameters defined in the AgentConfiguration class,
allowing for flexibility in model selection, retrieval methods, and system prompts.

Key Features:
- Intelligent query classification and routing
- Multi-step research planning for complex queries
- Integration with various retrieval providers (e.g., Elastic, Pinecone, MongoDB)
- Customizable language models for query analysis, research planning, and response generation
- Stateful conversation management for context-aware interactions

Usage:
    The main entry point for using this system is the `graph` object exported from this module.
    It can be invoked to process user inputs, conduct research , and generate
    informed responses based on retrieved information and conversation context.

For detailed configuration options and usage instructions, refer to the AgentConfiguration class
and individual component documentation within the retrieval_graph package.
"""  # noqa
