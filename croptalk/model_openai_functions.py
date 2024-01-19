from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
import os
from operator import itemgetter
from typing import Any

from langchain_core.documents import Document
from croptalk.retriever import format_docs
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder)
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.schema.runnable import (RunnablePassthrough)
from croptalk.retriever import RetrieverInput
from langchain.tools import StructuredTool
from chromadb.utils import embedding_functions
import chromadb
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.globals import set_debug

from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from croptalk.tools import tools

from dotenv import load_dotenv
load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')

set_debug(True)


# Vectorstore
vectorestore_dir = '/Users/katerina/Documents/CropGuard/Dec_demo/vectorstore'
emb_fn = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.PersistentClient(path=vectorestore_dir)
collection = client.get_collection(name="Demo", embedding_function=emb_fn)

# Tools


def format_chromadb_docs(result):
    """Formats the result of the ChromaDB query."""

    documents = result['documents'][0]
    metadatas = result['metadatas'][0]

    # Creating the new format
    docs = []
    for i in range(len(documents)):
        docs.append(Document(page_content=documents[i], metadata=metadatas[i]))

    return docs


def format_docs(docs) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i+1}' title={doc.metadata['title']}, page_id={doc.metadata['page']} doc_category={doc.metadata['doc_category']}, url={doc.metadata['source']}>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return formatted_docs


def query_chromadb(query):
    """Searches and returns information given the filters."""
    if not isinstance(query, str):
        raise ValueError(f"Query must be a string. Received: {query}")
    query_embedding = emb_fn([query])
    result = collection.query(query_embedding)
    docs = format_chromadb_docs(result)
    formatted_docs = format_docs(docs)
    return formatted_docs


find_docs = StructuredTool.from_function(
    name="find_docs",
    description="Searches and returns information given the filters.",
    func=query_chromadb,
    args_schema=RetrieverInput,
)

tools = [find_docs]

# Agent


def add_empty_doc(*args, **kwargs):
    kwargs["FindDocs"] = Document(page_content="Test doc", metadata={
        "title": "Test Name", "source": "https://www.google.com"})
    return kwargs


def initialize_agent_executor(model_name, tools,
                              memory_key="chat_history", input_key="question", output_key="output",
                              memory_model="gpt-3.5-turbo-1106"):

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{question}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    llm = ChatOpenAI(model=model_name, streaming=True, temperature=0.0)
    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )
    agent = (
        RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            chat_history=lambda x: x.get("chat_history", []),
            input=itemgetter(input_key)
        )
        | RunnablePassthrough.assign(
            context=lambda x: add_empty_doc(x)
        ).with_config(run_name="FindDocs")
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )
    memory_llm = ChatOpenAI(model=memory_model, temperature=0)
    memory = AgentTokenBufferMemory(
        memory_key=memory_key, llm=memory_llm, max_token_limit=6000)

    # retr = RunnableLambda(lambda x: query_chromadb(x[input_key]))
    agent_executor = (
        # RunnablePassthrough(question=retr.with_config(run_name="FindDocs"))
        # |
        AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            max_iterations=10,
            return_intermediate_steps=True,
        ) | itemgetter(output_key)
    ).with_config(run_name="AgentExecutor")

    return agent_executor


model_name = os.getenv("MODEL_NAME")
model = initialize_agent_executor(model_name=model_name, tools=tools)
