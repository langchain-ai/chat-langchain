from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from operator import itemgetter
from typing import Any, Callable, Dict, Optional, List

from langchain_core.documents import Document
from langchain_core.prompts import (ChatPromptTemplate, MessagesPlaceholder)
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.schema.runnable import RunnablePassthrough
from croptalk.chromadb_utils import create_chroma_filter

from langchain.tools import StructuredTool
from chromadb.utils import embedding_functions
import chromadb
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor


class ModelFactory:  # TODO: find a better name?

    def __init__(
        self,
        llm_model_name: str,
        vectorestore_dir: str,
        collection_name: str,
        embedding_function: Optional[Callable] = None,  # TODO: specify ins/outs of Callable
        memory_key: str = "chat_history",
        input_key: str ="question",
        output_key: str ="output",
    ) -> None:
        """
        Args:
            TODO
        """
        self.llm_model_name = llm_model_name
        self.vectorestore_dir = vectorestore_dir
        self.collection_name = collection_name
        self.embedding_function = embedding_function or embedding_functions.DefaultEmbeddingFunction()
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.collection = self._get_chromadb_collection()  # TODO: is it the right place?

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

    def _get_chromadb_collection(self):  # TODO: return type
        """
        Returns:
            TODO
        """
        chroma_client = chromadb.PersistentClient(path=self.vectorestore_dir)
        collection = chroma_client.get_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
        )
        return collection

    def _get_tools(self):  # TODO: return type
        """
        Returns:
            TODO
        """
        doc_retriever_tool = self._get_doc_retriever_tool()
        return [
            doc_retriever_tool,
        ]

    def _get_doc_retriever_tool(self):  # TODO: return type
        """
        Returns:
            TODO
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
        commodity: Optional[str] = Field(
            description="Commodity name. Example: Apples")
        state: Optional[str] = Field(description="State name. Example: California")
        county: Optional[str] = Field(description="County name. Example: Ventura")

    def _retriever_with_filter(
        self,
        query: str,
        doc_category: str = None,
        commodity: str = None,
        county: str = None,
        state: str = None,
        **kwargs,
    ) -> List[Document]:
        """
        TODO
        """
        """Retriever wrapper that allows to create chromadb where_filter and filter documents by there metadata."""
        if not isinstance(query, str):
            raise ValueError(f"Query must be a string. Received: {query}")

        where_filter = create_chroma_filter(
            commodity=commodity,
            county=county,
            state=state,
            doc_category=doc_category,
            include_common_docs=True,
        )

        return self._query_chromadb(query, where_filter=where_filter)

    def _query_chromadb(
        self,
        query: str,
        where_filter: Dict[str, Any] = None,
        k: int = 5,
    ):  # TODO: return type
        """
        Returns:
            TODO
        """
        """Searches and returns information given the filters."""

        query_embedding = self.embedding_function([query])
        result = self.collection.query(query_embedding, n_results=k, where=where_filter)
        docs = self._format_chromadb_docs(result)
        formatted_docs = self._format_docs(docs)
        return formatted_docs

    @staticmethod
    def _format_chromadb_docs(result):  # TODO: return type
        """
        Returns:
            TODO
        """
        """Formats the result of the ChromaDB query."""

        documents = result['documents'][0]
        metadatas = result['metadatas'][0]

        # Creating the new format
        docs = []
        for i in range(len(documents)):
            docs.append(Document(page_content=documents[i], metadata=metadatas[i]))

        return docs

    @staticmethod
    def _format_docs(docs) -> List[str]:
        """
        Returns:
            TODO
        """
        formatted_docs = []
        for i, doc in enumerate(docs):
            doc_string = f"<doc id='{i+1}' title={doc.metadata['title']}, page_id={doc.metadata['page']} doc_category={doc.metadata['doc_category']}, url={doc.metadata['source']}>{doc.page_content}</doc>"
            formatted_docs.append(doc_string)
        return formatted_docs
