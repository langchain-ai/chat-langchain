import os
from _operator import itemgetter
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain.globals import set_debug
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder)
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.runnable import (RunnableBranch,
                                       RunnableMap)
from langchain.tools.render import render_text_description
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from pydantic.v1 import BaseModel

from croptalk.document_retriever import DocumentRetriever
from croptalk.prompt_tools import TOOL_PROMPT
from croptalk.prompts_llm import COMMODITY_TEMPLATE, STATE_TEMPLATE, COUNTY_TEMPLATE, DOC_CATEGORY_TEMPLATE, \
    RESPONSE_TEMPLATE, REPHRASE_TEMPLATE, ROUTE_TEMPLATE
from croptalk.tools import tools
from croptalk.utils import initialize_llm

RENDERED_TOOLS = render_text_description(tools)
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
    DOC_CATEGORY_PROMPT = PromptTemplate.from_template(DOC_CATEGORY_TEMPLATE)

    condense_branch = create_condense_branch(llm)
    commodity_chain = (COMMODITY_PROMPT | llm | StrOutputParser()).with_config(
        run_name="IndentifyCommodity")
    state_chain = (STATE_PROMPT | llm | StrOutputParser()
                   ).with_config(run_name="IndentifyState")
    county_chain = (COUNTY_PROMPT | llm | StrOutputParser()
                    ).with_config(run_name="IndentifyCounty")
    doc_category_chain = (
            DOC_CATEGORY_PROMPT | llm | StrOutputParser()
    ).with_config(run_name="IndentifyDocCategory")

    retriever_func = RunnableLambda(lambda x: document_retriever.get_documents(query=x["question"],
                                                                               commodity=x['commodity'],
                                                                               state=x['state'],
                                                                               county=x['county'],
                                                                               doc_category=x['doc_category'],
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
                doc_category=doc_category_chain,
                question=itemgetter("question")
            ).with_config(run_name="CommodityChain")
            | retriever_func.with_config(run_name="FindDocs")
    )


def create_tool_chain(llm):
    def tool_pipe(model_output):
        try:
            tool_map = {tool.name: tool for tool in TOOLS}
            chosen_tool = tool_map[model_output["name"]]
            return itemgetter("arguments") | chosen_tool
        except Exception as e:
            return "NO_TOOL_ANSWER"

    def return_empty_str(output):
        return ""

    def tool_output(output):
        return output["output"]

    tool_prompt = PromptTemplate.from_template(TOOL_PROMPT)

    tool_chain = tool_prompt | llm | JsonOutputParser().with_config(
        run_name="ToolInput") | tool_pipe | StrOutputParser().with_config(run_name="ToolOutput")
    no_answer_chain = RunnableLambda(return_empty_str) | StrOutputParser()
    answer_chain = RunnableLambda(tool_output) | StrOutputParser()

    def route_tool_answer(tool_output):
        if "NO_TOOL_ANSWER" in tool_output["output"]:
            return no_answer_chain
        else:
            return answer_chain

    return (
            {"output": tool_chain, "question": itemgetter("question")}
            | RunnableLambda(route_tool_answer)
    )


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
    augmented_retrieval_chain = create_retriever_chain(basic_llm, document_retriever)
    tool_chain = create_tool_chain(basic_llm)

    chain = (
            PromptTemplate.from_template(ROUTE_TEMPLATE) | answer_llm | StrOutputParser()
    )

    def route(info):
        if "tools" in info["topic"].lower():
            return tool_chain
        else:
            return augmented_retrieval_chain

    route_tool_chain = {"topic": chain, "question": lambda x: x["question"]} | RunnableLambda(
        route
    )

    _context = RunnableMap(
        {
            "context_retriever": route_tool_chain,
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
                )
            }
            | _context
            | response_synthesizer
    )


model_name = os.getenv("MODEL_NAME")

collection_name = os.getenv("VECTORSTORE_COLLECTION")
doc_retriever = DocumentRetriever(collection_name=collection_name)

basic_llm = initialize_llm(model_name)
answer_llm = initialize_llm(model_name)

# Initialize the answer_chain
model = create_chain(basic_llm, answer_llm, doc_retriever)
