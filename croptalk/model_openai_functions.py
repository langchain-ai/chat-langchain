from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from operator import itemgetter
import os
from typing import Any, Callable, Dict, Optional, List

from langchain_core.documents import Document
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder)
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.schema.runnable import RunnablePassthrough
from croptalk.chromadb_utils import create_chroma_filter, get_chroma_collection

from langchain.tools import StructuredTool
from chromadb.api.types import QueryResult
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

from dotenv import load_dotenv
load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')


class OpenAIAgentModelFactory:
    """
    Class responsible for the creation of an OpenAI LLM chat agent.
    """

    def __init__(
        self,
        llm_model_name: str,
        vectorestore_dir: str,
        collection_name: str,
        top_k: int,
        embedding_function: Optional[Callable] = None,
        memory_key: str = "chat_history",
        input_key: str ="question",
        output_key: str ="output",
    ) -> None:
        """
        Args:
            llm_model_name: name of LLM model to use
            vectorestore_dir: directory where ChromaDB vectorstore files are located
            collection_name: collection name
            top_k: number of retrieved documents we are aiming for (i.e. top k)
            embedding_function: embedding function to use,
                                ChromaDB's default embedding function will be used if none is
                                provided
            memory_key: key to use to get chat history in chat messages
            input_key: key to use to get input (i.e. query) in chat messages
            output_key: key to use to get output (i.e. response) in chat messages
        """
        self.llm_model_name = llm_model_name
        self.vectorestore_dir = vectorestore_dir
        self.collection_name = collection_name
        self.top_k = top_k
        self.embedding_function = embedding_function or DefaultEmbeddingFunction()
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key

        self.collection = get_chroma_collection(
            vectorestore_dir=self.vectorestore_dir,
            collection_name=collection_name,
            embedding_function=self.embedding_function,
        )

    def get_model(self):  # TODO: return type
        """
        Returns:
            TODO
        """
        # TODO: save it in self and create it only the first time? or each time?
        # TODO: make the class callable __call__?
        tools = self._get_tools()
        llm = ChatOpenAI(model=self.llm_model_name, streaming=True, temperature=0.0)
        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in tools]
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a useful crop insurance assistant that provides accurate results based on retrieved docs.
                ALWAYS cite the relevant sources using source url, title, and page number."""),
                MessagesPlaceholder("chat_history", optional=True),  # TODO: use memory_key right?
                ("human", "{question}"),  # TODO: use input_key right?
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                chat_history=lambda x: x.get("chat_history", []),  # TODO: use memory_key right?
                input=itemgetter(self.input_key)
            )
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        memory_llm = ChatOpenAI(model=self.llm_model_name, temperature=0)  # TODO: use separate memory vs main model name like before?
        memory = AgentTokenBufferMemory(
            memory_key=self.memory_key, llm=memory_llm, max_token_limit=6000,
        )

        agent_executor = (
            AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=True,
                max_iterations=10,
                return_intermediate_steps=True,
            ) | itemgetter(self.output_key)
        ).with_config(run_name="AgentExecutor")

        return agent_executor

    def _get_tools(self) -> List[StructuredTool]:
        """
        Returns:
            a list of StructuredTools to provide to chat agent
        """
        doc_retriever_tool = self._get_doc_retriever_tool()
        return [
            doc_retriever_tool,
        ]

    def _get_doc_retriever_tool(self) -> StructuredTool:
        """
        Returns:
            a StructuredTool for document retrieval in chromaDB
        """
        find_docs = StructuredTool.from_function(
            name="FindDocs",
            description="Searches and returns information given the filters.",
            func=self._retriever_with_filter,
            args_schema=self._RetrieverInput,
        )
        return find_docs

    class _RetrieverInput(BaseModel):
        """Input schema for an llm-toolkit retriever."""
        query: str = Field(description="Query to look up in retriever")
        commodity: Optional[str] = Field(description="Commodity name. Example: Apples")
        state: Optional[str] = Field(description="State name. Example: California")
        county: Optional[str] = Field(description="County name. Example: Ventura")

    def _retriever_with_filter(
        self,
        query: str,
        doc_category: Optional[str] = None,
        commodity: Optional[str] = None,
        county: Optional[str] = None,
        state: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """
        Retriever wrapper that allows to create chromadb where_filter and filter documents by their
        metadata.

        Args:
            query: query to use when searching chroma DB
            doc_category: document category to filter on
            commodity: commodity to filter on
            county: county to filter on
            state: state to filter on

        Returns:
            list of retrieved documents matching query and filters, with formatting transformations
            for chatbot consumption
        """
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string. Received: {query}")

        where_filter = create_chroma_filter(
            commodity=commodity,
            county=county,
            state=state,
            doc_category=doc_category,
            include_common_docs=True,
        )

        return self._query_chromadb(query, where_filter=where_filter, k=self.top_k)

    def _query_chromadb(
        self,
        query: str,
        where_filter: Optional[Dict[str, Any]] = None,
        k: int = 5,
    ) -> List[str]:
        """
        Searches and returns information given the filters.

        Args:
            query: query to use when searching chroma DB
            where_filter: filter to use along with the query
            k: number of retrieved documents we are aiming for (i.e. top k)

        Returns:
            list of retrieved documents matching query and filters, with formatting transformations
            for chatbot consumption
        """
        query_embedding = self.embedding_function([query])
        result = self.collection.query(query_embedding, n_results=k, where=where_filter)
        docs = self._format_chromadb_docs(result)
        formatted_docs = self._format_docs(docs)
        return formatted_docs

    @staticmethod
    def _format_chromadb_docs(result: QueryResult) -> List[Document]:
        """
        Formats the result of the ChromaDB query.

        Args:
            result: chroadb query result

        Returns:
            list of (retrieved) documents equivalent to provided query result
        """
        documents = result['documents'][0]
        metadatas = result['metadatas'][0]

        # Creating the new format
        docs = []
        for i in range(len(documents)):
            docs.append(Document(page_content=documents[i], metadata=metadatas[i]))

        return docs

    @staticmethod
    def _format_docs(docs: List[Document]) -> List[str]:
        """
        Args:
            list of retrieved documents

        Returns:
            list of provided documents, with formatting transformations for chatbot consumption
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            doc_string = f"<doc id='{i+1}' title={doc.metadata['title']}, page_id={doc.metadata['page']} doc_category={doc.metadata['doc_category']}, url={doc.metadata['source']}>{doc.page_content}</doc>"
            formatted_docs.append(doc_string)
        return formatted_docs


# create singleton model
model_name = os.getenv("MODEL_NAME")
vectorestore_dir = os.getenv("VECTORSTORE_DIR")
collection_name = os.getenv("VECTORSTORE_COLLECTION")
top_k = int(os.getenv("VECTORSTORE_TOP_K"))
model = OpenAIAgentModelFactory(
    llm_model_name=model_name,
    vectorestore_dir=vectorestore_dir,
    collection_name=collection_name,
    top_k=top_k,
).get_model()
