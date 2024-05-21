import os
from typing import Annotated, Literal, Sequence, TypedDict

import weaviate
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, ToolMessage, convert_to_messages
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import (
    ConfigurableField,
    RunnableSequence,
    chain,
)
from langchain_openai import ChatOpenAI
from langchain_cohere import ChatCohere
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Weaviate
from langgraph.graph import StateGraph, START, END, add_messages

from backend.ingest import get_embeddings_model
from backend.constants import WEAVIATE_DOCS_INDEX_NAME


WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

COHERE_RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the sentence or paragraph that reference them - do not put them all at the end. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If there is nothing in the context relevant to the question at hand, just say "Hmm, \
I'm not sure." Don't try to make up an answer.

REMEMBER: If there is no relevant information within the context, just say "Hmm, I'm \
not sure." Don't try to make up an answer. Anything between the preceding 'context' \
html blocks is retrieved from a knowledge bank, not part of the conversation with the \
user.\
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def get_model(model_name: str) -> LanguageModelLike:
    gpt_3_5 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
    claude_3_sonnet = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0,
        max_tokens=4096,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", "not_provided"),
    )
    fireworks_mixtral = ChatFireworks(
        model="accounts/fireworks/models/mixtral-8x7b-instruct",
        temperature=0,
        max_tokens=16384,
        fireworks_api_key=os.environ.get("FIREWORKS_API_KEY", "not_provided"),
    )
    # TODO (VB): figure out why this is choking
    # gemini_pro = ChatGoogleGenerativeAI(
    #     model="gemini-pro",
    #     temperature=0,
    #     max_tokens=16384,
    #     convert_system_message_to_human=True,
    #     google_api_key=os.environ.get("GOOGLE_API_KEY", "not_provided"),
    # )
    cohere_command = ChatCohere(
        model="command",
        temperature=0,
        cohere_api_key=os.environ.get("COHERE_API_KEY", "not_provided"),
    )
    llm = gpt_3_5.configurable_alternatives(
        # This gives this field an id
        # When configuring the end runnable, we can then use this id to configure this field
        ConfigurableField(id="llm"),
        default_key="openai_gpt_3_5_turbo",
        anthropic_claude_3_sonnet=claude_3_sonnet,
        fireworks_mixtral=fireworks_mixtral,
        # google_gemini_pro=gemini_pro,
        cohere_command=cohere_command,
    ).with_fallbacks([
        gpt_3_5,
        claude_3_sonnet,
        fireworks_mixtral,
        #  gemini_pro,
        cohere_command
    ])
    return llm.with_config({"configurable": {"llm": model_name}})


def get_retriever() -> BaseRetriever:
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(),
        by_text=False,
        attributes=["source", "title"],
    )
    return weaviate_client.as_retriever(search_kwargs=dict(k=6))


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def retrieve_documents(state: AgentState):
    retriever = get_retriever()
    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    relevant_documents = retriever.get_relevant_documents(query)
    return {"messages": [ToolMessage(tool_call_id="fake_tool_call", content=format_docs(relevant_documents))]}


def retrieve_documents_with_chat_history(state: AgentState, config):
    model_name = config.get("configurable", {}).get("model_name", "anthropic_claude_3_sonnet")
    retriever = get_retriever()
    model = get_model(model_name)

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT |
        model | 
        StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )

    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    retriever_with_condensed_question = condense_question_chain | retriever
    relevant_documents = retriever_with_condensed_question.invoke({"question": query, "chat_history": get_chat_history(state)})
    return {"messages": [ToolMessage(tool_call_id="fake_tool_call", content=format_docs(relevant_documents))]}


def route_to_retriever(state: AgentState) -> Literal["retriever", "retriever_with_chat_history"]:
    if len(state["messages"]) == 1:
        return "retriever"
    else:
        return "retriever_with_chat_history"


def get_chat_history(state: AgentState):
    chat_history = []
    for message in convert_to_messages(state["messages"]):
        if isinstance(message, AIMessage) or isinstance(message, HumanMessage):
            chat_history.append(message)
    return chat_history


def get_last_message(messages, condition):
    for message in reversed(messages):
        if condition(message):
            return message

    return None


def synthesize_response(state: AgentState, config):
    messages = convert_to_messages(state["messages"])
    model_name = config.get("configurable", {}).get("model_name", "anthropic_claude_3_sonnet")
    model = get_model(model_name)

    last_human_message = get_last_message(messages, lambda message: isinstance(message, HumanMessage))
    if last_human_message is None:
        raise AssertionError("Expecting to have at least 1 human message")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    default_response_synthesizer = prompt | model
    cohere_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COHERE_RESPONSE_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # TODO (VB): uncommend this, commented for debugging

    # @chain
    # def cohere_response_synthesizer(input: dict) -> RunnableSequence:
    #     return cohere_prompt | model.bind(source_documents=input["docs"])

    # response_synthesizer = (
    #     default_response_synthesizer.configurable_alternatives(
    #         ConfigurableField("llm"),
    #         default_key="openai_gpt_3_5_turbo",
    #         anthropic_claude_3_sonnet=default_response_synthesizer,
    #         fireworks_mixtral=default_response_synthesizer,
    #         google_gemini_pro=default_response_synthesizer,
    #         cohere_command=cohere_response_synthesizer,
    #     )
    # ).with_config(run_name="GenerateResponse")
    synthesized_response = default_response_synthesizer.invoke({
        "question": last_human_message.content,
        "context": messages[-1].content,
        "chat_history": get_chat_history(state)
    })
    return {"messages": [synthesized_response]}


workflow = StateGraph(AgentState)

workflow.add_node("response_synthesizer", synthesize_response)
workflow.add_node("retriever", retrieve_documents)
workflow.add_node("retriever_with_chat_history", retrieve_documents_with_chat_history)

workflow.add_conditional_edges(START, route_to_retriever)
workflow.add_edge("retriever", "response_synthesizer")
workflow.add_edge("retriever_with_chat_history", "response_synthesizer")
workflow.add_edge("response_synthesizer", END)

graph = workflow.compile()