from dotenv import load_dotenv
import os
from operator import itemgetter
from typing import Dict, List, Optional
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import (Runnable, RunnableBranch,
                                       RunnableLambda, RunnableMap, RunnableParallel)

from langchain.globals import set_debug
from pydantic.v1 import BaseModel
from croptalk.prompts_llm import RESPONSE_TEMPLATE, REPHRASE_TEMPLATE, COMMODITY_TEMPLATE, STATE_TEMPLATE, COUNTY_TEMPLATE, INS_PLAN_TEMPLATE, DOC_CATEGORY_TEMPLATE
from croptalk.document_retriever import DocumentRetriever

set_debug(True)

load_dotenv('secrets/.env.secret')
load_dotenv('secrets/.env.shared')


def create_retriever_chain(llm: BaseLanguageModel, document_retriever: DocumentRetriever) -> Runnable:

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_chain_hist = (CONDENSE_QUESTION_PROMPT | llm | StrOutputParser(
    )).with_config(run_name="CondenseQuestion")
    condense_chain_no_hist = RunnableLambda(itemgetter("question")).with_config(
        run_name="RetrievalChainWithNoHistory")

    condense_branch = RunnableBranch(
        (
            RunnableLambda(lambda x: not bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"),
            condense_chain_no_hist,
        ),
        (condense_chain_hist),
    ).with_config(run_name="RouteDependingOnChatHistory")

    COMMODITY_PROMPT = PromptTemplate.from_template(COMMODITY_TEMPLATE)
    STATE_PROMPT = PromptTemplate.from_template(STATE_TEMPLATE)
    COUNTY_PROMPT = PromptTemplate.from_template(COUNTY_TEMPLATE)
    INS_PLAN_PROMPT = PromptTemplate.from_template(INS_PLAN_TEMPLATE)
    DOC_CATEGORY_PROMPT = PromptTemplate.from_template(DOC_CATEGORY_TEMPLATE)

    commodity_chain = (COMMODITY_PROMPT | llm | StrOutputParser()).with_config(
        run_name="IndentifyCommodity")
    state_chain = (STATE_PROMPT | llm | StrOutputParser()
                   ).with_config(run_name="IndentifyState")
    county_chain = (COUNTY_PROMPT | llm | StrOutputParser()
                    ).with_config(run_name="IndentifyCounty")
    ins_plan_chain = (INS_PLAN_PROMPT | llm | StrOutputParser()
                      ).with_config(run_name="IndentifyPlan")
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
            insurance_plan=ins_plan_chain,
            doc_category=doc_category_chain,
            question=itemgetter("question")
        ).with_config(run_name="CommodityChain")
        | retriever_func.with_config(run_name="FindDocs")
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

    retriever_chain = create_retriever_chain(basic_llm, document_retriever)

    _context = RunnableMap(
        {
            "context": retriever_chain,
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
