import contextlib
import os
from collections import defaultdict
from typing import Annotated, Iterator, Literal, Optional, Sequence, TypedDict

import weaviate
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    convert_to_messages,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import ConfigurableField, RunnableConfig, ensure_config
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_weaviate import WeaviateVectorStore
from langgraph.graph import END, StateGraph, add_messages
from langsmith import Client as LangsmithClient

from backend.constants import WEAVIATE_DOCS_INDEX_NAME
from backend.ingest import get_embeddings_model

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

RESPONSE_TEMPLATE = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based on the provided search results (URL and content). You must \
only use information from the provided search results, which may include both internal \
and external sources. Use an unbiased and journalistic tone. Combine search results \
together into a coherent answer. Do not repeat text. Cite search results using [${{number}}] \
notation. Only cite the most relevant results that answer the question accurately. \
Place these citations at the end of the sentence or paragraph that reference them - do \
not put them all at the end.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end.

If using information from external sources, clearly state this in your answer.

Anything between the following `context` html blocks is retrieved from a knowledge \
bank or external sources, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: Only use information from the provided context. If using external sources, \
clearly state this in your answer. If the information doesn't directly answer the \
question, acknowledge this while providing the most relevant details available.\
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


OPENAI_MODEL_KEY = "openai_gpt_4o_mini"
ANTHROPIC_MODEL_KEY = "anthropic_claude_3_haiku"
FIREWORKS_MIXTRAL_MODEL_KEY = "fireworks_mixtral"
GOOGLE_MODEL_KEY = "google_gemini_pro"
COHERE_MODEL_KEY = "cohere_command"
GROQ_LLAMA_3_MODEL_KEY = "groq_llama_3"
# Not exposed in the UI
GPT_4O_MODEL_KEY = "openai_gpt_4o"
CLAUDE_35_SONNET_MODEL_KEY = "anthropic_claude_3_5_sonnet"

FEEDBACK_KEYS = ["user_score", "user_click"]


def update_documents(
    _: list[Document], right: list[Document] | list[dict]
) -> list[Document]:
    res: list[Document] = []

    for item in right:
        if isinstance(item, dict):
            res.append(Document(**item))
        elif isinstance(item, Document):
            res.append(item)
        else:
            raise TypeError(f"Got unknown document type '{type(item)}'")
    return res


class AgentState(TypedDict):
    query: str
    documents: Annotated[list[Document], update_documents]
    messages: Annotated[list[AnyMessage], add_messages]
    # for convenience in evaluations
    answer: str
    feedback_urls: dict[str, list[str]]


gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0, streaming=True)

claude_3_haiku = ChatAnthropic(
    model="claude-3-haiku-20240307",
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
gemini_pro = ChatGoogleGenerativeAI(
    model="gemini-pro",
    temperature=0,
    max_output_tokens=16384,
    convert_system_message_to_human=True,
    google_api_key=os.environ.get("GOOGLE_API_KEY", "not_provided"),
)
cohere_command = ChatCohere(
    model="command",
    temperature=0,
    cohere_api_key=os.environ.get("COHERE_API_KEY", "not_provided"),
)
groq_llama3 = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    groq_api_key=os.environ.get("GROQ_API_KEY", "not_provided"),
)

# Not exposed in the UI
gpt_4o = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.3, streaming=True)
claude_35_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    temperature=0.7,
)

llm = gpt_4o_mini.configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="model_name"),
    default_key=OPENAI_MODEL_KEY,
    **{
        ANTHROPIC_MODEL_KEY: claude_3_haiku,
        FIREWORKS_MIXTRAL_MODEL_KEY: fireworks_mixtral,
        GOOGLE_MODEL_KEY: gemini_pro,
        COHERE_MODEL_KEY: cohere_command,
        GROQ_LLAMA_3_MODEL_KEY: groq_llama3,
        GPT_4O_MODEL_KEY: gpt_4o,
        CLAUDE_35_SONNET_MODEL_KEY: claude_35_sonnet,
    },
).with_fallbacks(
    [
        gpt_4o_mini,
        claude_3_haiku,
        fireworks_mixtral,
        gemini_pro,
        cohere_command,
        groq_llama3,
    ]
)


@contextlib.contextmanager
def get_retriever(k: Optional[int] = None) -> Iterator[BaseRetriever]:
    with weaviate.connect_to_weaviate_cloud(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.classes.init.Auth.api_key(
            os.environ.get("WEAVIATE_API_KEY", "not_provided")
        ),
        skip_init_checks=True,
    ) as weaviate_client:
        store = WeaviateVectorStore(
            client=weaviate_client,
            index_name=WEAVIATE_DOCS_INDEX_NAME,
            text_key="text",
            embedding=get_embeddings_model(),
            attributes=["source", "title"],
        )
        k = k or 6
        yield store.as_retriever(search_kwargs=dict(k=k))


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def retrieve_documents(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    config = ensure_config(config)
    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    with get_retriever(k=config["configurable"].get("k")) as retriever:
        relevant_documents = retriever.invoke(query)
    return {"query": query, "documents": relevant_documents}


def retrieve_documents_with_chat_history(
    state: AgentState, *, config: Optional[RunnableConfig] = None
) -> AgentState:
    config = ensure_config(config)
    model = llm.with_config(tags=["nostream"])

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | model | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )

    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    with get_retriever(k=config["configurable"].get("k")) as retriever:
        retriever_with_condensed_question = condense_question_chain | retriever
        # NOTE: we're ignoring the last message here, as it's going to contain the most recent
        # query and we don't want that to be included in the chat history
        relevant_documents = retriever_with_condensed_question.invoke(
            {"question": query, "chat_history": get_chat_history(messages[:-1])}
        )
    return {"query": query, "documents": relevant_documents}




def perform_external_search(state: AgentState) -> AgentState:
    search = DuckDuckGoSearchAPIWrapper()
    results = search.results(state["query"], num_results=3)
    
    external_docs = [
        Document(
            page_content=result["snippet"],
            metadata={"source": result["link"], "title": result["title"]}
        )
        for result in results
    ]
    
    # Combine the original and external search documents
    all_documents = state["documents"] + external_docs
    
    state["documents"] = all_documents
    return state

def check_relevance(state: AgentState) -> Literal["synthesize", "external_search"]:
    # Simple relevance check
    if not state["documents"]:
        return "external_search"
    
    query_keywords = set(state["query"].lower().split())
    for doc in state["documents"]:
        if any(keyword in doc.page_content.lower() for keyword in query_keywords):
            return "synthesize"
    return "external_search"

def route_to_relevance_check(state: AgentState) -> Literal["relevance_check"]:
    return "relevance_check"


def route_to_retriever(
    state: AgentState,
) -> Literal["retriever", "retriever_with_chat_history"]:
    # at this point in the graph execution there is exactly one (i.e. first) message from the user,
    # so use basic retriever without chat history
    if len(state["messages"]) == 1:
        return "retriever"
    else:
        return "retriever_with_chat_history"


def get_chat_history(messages: Sequence[BaseMessage]) -> Sequence[BaseMessage]:
    chat_history = []
    for message in messages:
        if (isinstance(message, AIMessage) and not message.tool_calls) or isinstance(
            message, HumanMessage
        ):
            chat_history.append({"content": message.content, "role": message.type})
    return chat_history


def get_feedback_urls(config: RunnableConfig) -> dict[str, list[str]]:
    ls_client = LangsmithClient()
    run_id = config["configurable"].get("run_id")
    if run_id is None:
        return {}

    tokens = ls_client.create_presigned_feedback_tokens(run_id, FEEDBACK_KEYS)
    key_to_token_urls = defaultdict(list)

    for token_idx, token in enumerate(tokens):
        key_idx = token_idx % len(FEEDBACK_KEYS)
        key = FEEDBACK_KEYS[key_idx]
        key_to_token_urls[key].append(token.url)
    return key_to_token_urls


def synthesize_response(
    state: AgentState,
    config: RunnableConfig,
    model: LanguageModelLike,
    prompt_template: str,
) -> AgentState:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", prompt_template),
            ("placeholder", "{chat_history}"),
            ("human", "{question}"),
        ]
    )
    response_synthesizer = prompt | model
    synthesized_response = response_synthesizer.invoke(
        {
            "question": state["query"],
            "context": format_docs(state["documents"]),
            # NOTE: we're ignoring the last message here, as it's going to contain the most recent
            # query and we don't want that to be included in the chat history
            "chat_history": get_chat_history(
                convert_to_messages(state["messages"][:-1])
            ),
        }
    )
    # finally, add feedback URLs so that users can leave feedback
    feedback_urls = get_feedback_urls(config)
    return {
        "messages": [synthesized_response],
        "answer": synthesized_response.content,
        "feedback_urls": feedback_urls,
    }


def synthesize_response_default(
    state: AgentState, config: RunnableConfig
) -> AgentState:
    return synthesize_response(state, config, llm, RESPONSE_TEMPLATE)


def synthesize_response_cohere(state: AgentState, config: RunnableConfig) -> AgentState:
    model = llm.bind(documents=state["documents"])
    return synthesize_response(state, config, model, COHERE_RESPONSE_TEMPLATE)


def route_to_response_synthesizer(
    state: AgentState, config: RunnableConfig
) -> Literal["response_synthesizer", "response_synthesizer_cohere"]:
    model_name = config.get("configurable", {}).get("model_name", OPENAI_MODEL_KEY)
    if model_name == COHERE_MODEL_KEY:
        return "response_synthesizer_cohere"
    else:
        return "response_synthesizer"


class Configuration(TypedDict):
    model_name: str
    k: int


class InputSchema(TypedDict):
    messages: list[AnyMessage]


workflow = StateGraph(AgentState, Configuration, input=InputSchema)

# define nodes
workflow.add_node("retriever", retrieve_documents)
workflow.add_node("retriever_with_chat_history", retrieve_documents_with_chat_history)
workflow.add_node("response_synthesizer", synthesize_response_default)
workflow.add_node("response_synthesizer_cohere", synthesize_response_cohere)


# Add new nodes
workflow.add_node("relevance_check", check_relevance)
workflow.add_node("external_search", perform_external_search)



# set entry point to retrievers
workflow.set_conditional_entry_point(route_to_retriever)


# Modify existing edges and add new ones
workflow.add_conditional_edges("retriever", route_to_relevance_check)
workflow.add_conditional_edges("retriever_with_chat_history", route_to_relevance_check)
workflow.add_edge("relevance_check", "response_synthesizer", edge_condition="synthesize")
workflow.add_edge("relevance_check", "external_search", edge_condition="external_search")
workflow.add_edge("external_search", "response_synthesizer")



# connect synthesizers to terminal node
workflow.add_edge("response_synthesizer", END)
workflow.add_edge("response_synthesizer_cohere", END)

graph = workflow.compile()
