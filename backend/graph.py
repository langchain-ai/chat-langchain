import os
from collections import defaultdict
from typing import Annotated, Literal, Optional, Sequence, TypedDict

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


def get_retriever(k: Optional[int] = None) -> BaseRetriever:
    weaviate_client = weaviate.connect_to_wcs(
        cluster_url=os.environ["WEAVIATE_URL"],
        auth_credentials=weaviate.classes.init.Auth.api_key(
            os.environ.get("WEAVIATE_API_KEY", "not_provided")
        ),
        skip_init_checks=True,
    )
    weaviate_client = WeaviateVectorStore(
        client=weaviate_client,
        index_name=WEAVIATE_DOCS_INDEX_NAME,
        text_key="text",
        embedding=get_embeddings_model(),
        attributes=["source", "title"],
    )
    k = k or 6
    return weaviate_client.as_retriever(search_kwargs=dict(k=k))


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
    retriever = get_retriever(k=config["configurable"].get("k"))
    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    relevant_documents = retriever.invoke(query)
    return {"query": query, "documents": relevant_documents}


def retrieve_documents_with_chat_history(state: AgentState) -> AgentState:
    retriever = get_retriever()
    model = llm.with_config(tags=["nostream"])

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | model | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )

    messages = convert_to_messages(state["messages"])
    query = messages[-1].content
    retriever_with_condensed_question = condense_question_chain | retriever
    # NOTE: we're ignoring the last message here, as it's going to contain the most recent
    # query and we don't want that to be included in the chat history
    relevant_documents = retriever_with_condensed_question.invoke(
        {"question": query, "chat_history": get_chat_history(messages[:-1])}
    )
    return {"query": query, "documents": relevant_documents}


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

# set entry point to retrievers
workflow.set_conditional_entry_point(route_to_retriever)

# connect retrievers and response synthesizers
workflow.add_conditional_edges("retriever", route_to_response_synthesizer)
workflow.add_conditional_edges(
    "retriever_with_chat_history", route_to_response_synthesizer
)

# connect synthesizers to terminal node
workflow.add_edge("response_synthesizer", END)
workflow.add_edge("response_synthesizer_cohere", END)

graph = workflow.compile()

from langgraph.graph import MessagesState


class ResearcherState(TypedDict):
    sub_question: str
    queries: list[str]
    documents: Annotated[list[Document], update_documents]


class AgentState(MessagesState):
    sub_questions: list[str]
    documents: Annotated[list[Document], update_documents]


generate_queries_prompt = """Generate 3 search queries to search for \
to answer the user's question. These search queries should be diverse in nature - do not generate \
repetitive ones."""

def generate_queries(state: ResearcherState):
    messages = [
       {"role": "system", "content": generate_queries_prompt},
       {"role": "human", "content": state['sub_question']}
    ]

    class Response(TypedDict):
        queries: list[str]

    response = gpt_4o_mini.with_structured_output(Response).invoke(messages)
    return response

class QueryState(TypedDict):
    query: str

def query(state: QueryState):
    retriever = get_retriever()
    docs = retriever.invoke(state['query'])
    return {"documents": docs}


from langgraph.constants import Send
from langgraph.graph import END, START
def do_queries(state: ResearcherState) -> Literal['query']:
    return [Send("query", {"query": q}) for q in state['queries']]

researcher = StateGraph(ResearcherState)
researcher.add_node(query)
researcher.add_node(generate_queries)
researcher.add_edge(START, "generate_queries")
researcher.add_conditional_edges("generate_queries", do_queries)
researcher.add_edge("query", END)
researcher = researcher.compile()


generate_questions_prompt = """You are a LangChain expert, here to assist with any and all questions or issues with LangChain, LangGraph, LangSmith, or any related functionality. Users may come to you with questions or issues.

You are world class researcher. Based on the conversation below, you have the option to generate 3 research questions to research in the LangChain documentation to resolve the users question. These questions should be diverse and cover a spectrum of possibilities. If the question references multiple concepts, or seems like understanding multiple concepts may be needed in order to answer, you should generate a sub question for each of those concepts. Do not answer directly about any LangChain questions without calling the research tool

Because you will research these questions in the LangChain documentation, if you need any more information from the user ask them for that BEFORE calling this tool. You should respond asking for more information only when:

- The user complains about an error but doesnt provide the error 
- The user says something isn't working but doesnt explain why/how it's not working.

Otherwise, go ahead and research their question!"""
def generate_questions(state: AgentState):
    messages = [{"role": "system", "content": generate_questions_prompt}] + state['messages']

    class ResearchQuestions(TypedDict):
        """Ask research questions."""
        sub_questions: list[str]

    response = gpt_4o_mini.bind_tools([ResearchQuestions]).invoke(messages)
    if len(response.tool_calls) == 0:
        return {"messages": response}
    else:
        return response.tool_calls[0]['args']

def route_question(state: AgentState) -> Literal['generate', 'researcher', END]:
    if len(state.get("sub_questions", [])) > 0:
        return [Send("researcher", {"sub_question": q}) for q in state["sub_questions"][:1]]
    else:
        if isinstance(state['messages'][-1], HumanMessage):
            return "generate"
        else:
            return END

def remove_question(state):
    return {"sub_questions": state["sub_questions"][1:]}


RESPONSE_TEMPLATE1 = """\
You are an expert programmer and problem-solver, tasked with answering any question \
about Langchain.

Generate a comprehensive and informative answer for the \
given question based solely on the provided search results (URL and content). \
Do NOT ramble, and adjust your response length based on the question. If they ask \
a question that can be answered in one sentence, do that. If 5 paragraphs of detail is needed, \
do that. You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. Cite search results using [${{number}}] notation. Only cite the most \
relevant results that answer the question accurately. Place these citations at the end \
of the individual sentence or paragraph that reference them. \
Do not put them all at the end, but rather sprinkle them throughout. If \
different results refer to different entities within the same name, write separate \
answers for each entity.

You should use bullet points in your answer for readability. Put citations where they apply
rather than putting them all at the end. DO NOT PUT THEM ALL THAT END, PUT THEM IN THE BULLET POINTS.

If there is nothing in the context relevant to the question at hand, do NOT make up an answer. \
Rather, tell them why you're unsure and ask for any additional information that may help you answer better.

Anything between the following `context`  html blocks is retrieved from a knowledge \
bank, not part of the conversation with the user. 

<context>
    {context} 
<context/>
"""


def generate(state: AgentState):
    context = format_docs(state['documents'])
    prompt = RESPONSE_TEMPLATE1.format(context=context)
    response = gpt_4o_mini.invoke([{"role": "system", "content": prompt}] + state['messages'])
    return {"messages": response}

agent = StateGraph(AgentState, input=MessagesState, output=MessagesState)
agent.add_node("researcher", researcher)
agent.add_node(generate_questions)
agent.add_node(generate)
agent.add_node(remove_question)
agent.add_conditional_edges("generate_questions", route_question)
agent.add_edge("researcher", "remove_question")
agent.add_conditional_edges("remove_question", route_question)
agent.add_edge("generate", END)
agent.add_edge(START, "generate_questions")
agent = agent.compile()
