import os
from operator import itemgetter
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
)
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain.chains.base import Chain
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools import StructuredTool
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from croptalk.document_retriever import DocumentRetriever
from croptalk.tools import TOOLS

load_dotenv("secrets/.env.secret")
load_dotenv("secrets/.env.shared")


class OpenAIAgentModelFactory:
    """
    Class responsible for the creation of an OpenAI LLM chat agent.
    """

    def __init__(
        self,
        llm_model_name: str,
        document_retriever: DocumentRetriever,
        tools: List[StructuredTool],
        top_k: int,
        memory_key: str = "chat_history",
        input_key: str = "question",
        output_key: str = "output",
    ) -> None:
        """
        Args:
            llm_model_name: name of LLM model to use
            document_retriever: document retriever to use as agent's tool
            top_k: number of retrieved documents we are aiming for (i.e. top k)
            memory_key: key to use to get chat history in chat messages
            input_key: key to use to get input (i.e. query) in chat messages
            output_key: key to use to get output (i.e. response) in chat messages
        """
        self.llm_model_name = llm_model_name
        self.document_retriever = document_retriever
        self.tools = tools
        self.top_k = top_k
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key

    def get_model(self) -> Tuple[Chain, AgentTokenBufferMemory]:
        """
        Returns:
            - newly created OpenAI LLM chat agent using ChromaDB vectorstore
            - memory object
        """
        # create LLM
        llm = ChatOpenAI(model=self.llm_model_name, streaming=True, temperature=0.0)
        tools = self._get_tools()
        llm_with_tools = llm.bind(
            functions=[format_tool_to_openai_function(t) for t in tools]
        )

        # create chat chain
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant named CropTalk representing the company CropGuard.
                    CropGuard uses cutting-edge AI to help crop insurance agents and farmers select
                    optimal crop insurance policies tailored to growers' unique risk profiles.
                    You provide accurate results based on retrieved docs.
                    ALWAYS cite the relevant sources using this format : ( Document : title, page : page_id , url : link  )""",
                ),
                MessagesPlaceholder(self.memory_key, optional=True),
                ("human", f"{{{self.input_key}}}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        agent = (
            RunnablePassthrough.assign(
                agent_scratchpad=lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
                chat_history=lambda x: x.get(self.memory_key, []),
                input=itemgetter(self.input_key),
            )
            | prompt
            | llm_with_tools
            | OpenAIFunctionsAgentOutputParser()
        )
        memory_llm = ChatOpenAI(model=self.llm_model_name, temperature=0)
        memory = AgentTokenBufferMemory(
            memory_key=self.memory_key,
            llm=memory_llm,
            max_token_limit=6000,
        )

        agent_executor = (
            AgentExecutor(
                agent=agent,
                tools=tools,
                memory=memory,
                verbose=True,
                max_iterations=10,
                return_intermediate_steps=True,
            )
            | itemgetter(self.output_key)
        ).with_config(run_name="AgentExecutor")

        return agent_executor, memory

    def _get_tools(self) -> List[StructuredTool]:
        """
        Returns:
            a list of StructuredTools to provide to chat agent
        """
        doc_retriever_tool = self._get_doc_retriever_tool()
        return [
            doc_retriever_tool,
        ] + self.tools

    def _get_doc_retriever_tool(self) -> StructuredTool:
        """
        Returns:
            a StructuredTool for document retrieval in chromaDB
        """
        find_docs = StructuredTool.from_function(
            name="FindDocs",
            description="This tool is used to find information contained in the Crop Insurance Handbook (CIH),"
                        "in the Basic Provisions documents (BP) and within the Crop Provision documents (CP). "
                        "It should NOT be used to answer county specific insurance questions."
                        "It should NOT be used to retrieve insurance market data and statistics."
                        "\b"
                        "The Crop Insurance Handbook covers various topics such as policy provisions, procedures "
                        "for policy administration, standards for determining insurability, and requirements for "
                        "reporting and recordkeeping. "
                        "\b"
                        "The Basic Provisions include: Eligibility criteria for participation, requirements for "
                        "reporting, procedures for obtaining insurance coverage, provisions for determining coverage "
                        "levels, indemnity payments, and loss adjustments. Guidelines for compliance with program rules"
                        "\b"
                        "The Crop Provision include : Crop-specific coverage details, fates and deadlines, "
                        "crop-specific rules and practices, exclusions and limitations",
            func=lambda **kwargs: self.document_retriever.get_documents(**kwargs, top_k=self.top_k),
            args_schema=self._RetrieverInput,
        )
        return find_docs

    class _RetrieverInput(BaseModel):
        """Input schema for an llm-toolkit retriever."""

        query: str = Field(description="Query to look up in retriever")
        commodity: Optional[str] = Field(description="Commodity name. Example: Apples")
        state: Optional[str] = Field(description="State name. Example: California")
        county: Optional[str] = Field(description="County name. Example: Ventura")
        doc_category: Optional[str] = Field(description="Document category. Example: SP")


# create singleton model
model_name = os.getenv("MODEL_NAME")
collection_name = os.getenv("VECTORSTORE_COLLECTION")
top_k = int(os.getenv("VECTORSTORE_TOP_K"))
doc_retriever = DocumentRetriever(collection_name=collection_name)
model, memory = OpenAIAgentModelFactory(
    llm_model_name=model_name,
    document_retriever=doc_retriever,
    tools=TOOLS,
    top_k=top_k,
).get_model()
