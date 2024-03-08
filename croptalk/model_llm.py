import os
from operator import itemgetter
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_debug
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (Runnable, RunnableBranch,
                                       RunnableLambda, RunnableMap, RunnableParallel)

from langchain_core.output_parsers import JsonOutputParser
from pydantic.v1 import BaseModel

from croptalk.document_retriever import DocumentRetriever
from croptalk.prompts_llm import RESPONSE_TEMPLATE, REPHRASE_TEMPLATE, COMMODITY_TEMPLATE, STATE_TEMPLATE, \
    COUNTY_TEMPLATE, INS_PLAN_TEMPLATE
from croptalk.tools import tools

TOOLS = tools

set_debug(True)

load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')


def create_condense_branch(llm):
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_chain_hist = (CONDENSE_QUESTION_PROMPT | llm | StrOutputParser(
    )).with_config(run_name="CondenseQuestion")
    condense_chain_no_hist = RunnableLambda(itemgetter("question")).with_config(
        run_name="RetrievalChainWithNoHistory")

    return RunnableBranch(
        (
            RunnableLambda(lambda x: not bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"),
            condense_chain_no_hist,
        ),
        (condense_chain_hist),
    ).with_config(run_name="RouteDependingOnChatHistory")


def create_retriever_chain(llm: BaseLanguageModel, document_retriever: DocumentRetriever) -> Runnable:
    COMMODITY_PROMPT = PromptTemplate.from_template(COMMODITY_TEMPLATE)
    STATE_PROMPT = PromptTemplate.from_template(STATE_TEMPLATE)
    COUNTY_PROMPT = PromptTemplate.from_template(COUNTY_TEMPLATE)
    INS_PLAN_PROMPT = PromptTemplate.from_template(INS_PLAN_TEMPLATE)

    condense_branch = create_condense_branch(llm)
    commodity_chain = (COMMODITY_PROMPT | llm | StrOutputParser()).with_config(
        run_name="IndentifyCommodity")
    state_chain = (STATE_PROMPT | llm | StrOutputParser()
                   ).with_config(run_name="IndentifyState")
    county_chain = (COUNTY_PROMPT | llm | StrOutputParser()
                    ).with_config(run_name="IndentifyCounty")
    ins_plan_chain = (INS_PLAN_PROMPT | llm | StrOutputParser()
                      ).with_config(run_name="IndentifyPlan")

    retriever_func = RunnableLambda(lambda x: document_retriever.get_documents(query=x["question"],
                                                                               commodity=x['commodity'],
                                                                               state=x['state'],
                                                                               county=x['county'],
                                                                               include_common_docs=True)
                                    ).with_config(run_name="RetrieverWithFilter")

    return (
            RunnableParallel(
                question=condense_branch
            )
            |
            RunnableParallel(
                commodity=commodity_chain,
                state=state_chain,
                county=county_chain,
                insurance_plan=ins_plan_chain,
                question=itemgetter("question")
            ).with_config(run_name="CommodityChain")
            | retriever_func.with_config(run_name="FindDocs")
    )


def create_tool_chain(llm):

    system_prompt = """
    You are an assistant that has access to the following set of tools. 
    Here are the names and descriptions for each tool:

    {rendered_tools}

    Given the user questions, return the name and input of the tool to use. 
    Return your response as a JSON blob with 'name' and 'arguments' keys.

    Do not use tools if they are not necessary

    this is the question you are being asked : {question}
    
    """

    prompt_tool = PromptTemplate.from_template(system_prompt)

    def tool_chain(model_output):
        try:
            tool_map = {tool.name: tool for tool in TOOLS}
            chosen_tool = tool_map[model_output["name"]]
            return itemgetter("arguments") | chosen_tool
        except Exception as e:
            return ""

    tool_chain = prompt_tool | llm | JsonOutputParser() | tool_chain | StrOutputParser()

    prompt2 = PromptTemplate.from_template(
        "You are a helpful assistant. Answer the provided question : {question}. Knowing that the this answer was "
        "calculated using your own tool: {output}."
    )

    final_tool_chain = (
            {"output": tool_chain, "question": itemgetter("question")}
            | prompt2
            | llm
            | StrOutputParser()
    )
    return final_tool_chain





class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(
                HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(
        basic_llm: BaseLanguageModel,
        answer_llm: BaseLanguageModel,
        document_retriever: DocumentRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(basic_llm, document_retriever)
    tool_chain = create_tool_chain(basic_llm)

    _context = RunnableMap(
        {
            "context_retriever": retriever_chain,
            "context_tools": tool_chain,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),

        }
    ).with_config(run_name="RetrieveDocs")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    response_synthesizer = (prompt | answer_llm | StrOutputParser()).with_config(
        run_name="GenerateResponse",
    )
    return (
            {
                "question": RunnableLambda(itemgetter("question")).with_config(
                    run_name="Itemgetter:question"
                ),
                "chat_history": RunnableLambda(serialize_history).with_config(
                    run_name="SerializeHistory"
                ),
                "rendered_tools": RunnableLambda(itemgetter("rendered_tools")).with_config(
                    run_name="Itemgetter:rendered_tools"
                ),
            }
            | _context
            | response_synthesizer
    )


def initialize_llm(model):
    return ChatOpenAI(
        model=model,
        streaming=True,
        temperature=0,
    )


model_name = os.getenv("MODEL_NAME")

collection_name = os.getenv("VECTORSTORE_COLLECTION")
doc_retriever = DocumentRetriever(collection_name=collection_name)

basic_llm = initialize_llm(model_name)
answer_llm = initialize_llm(model_name)

# Initialize the answer_chain
model = create_chain(basic_llm, answer_llm, doc_retriever)
