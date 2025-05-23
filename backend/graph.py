import os
from typing import Annotated, Literal, Sequence, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import (
    AIMessage,
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
from langchain_core.runnables import ConfigurableField, RunnableConfig
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_milvus import Milvus
from langchain_ollama import ChatOllama
from pymilvus import connections
from langgraph.graph import END, StateGraph, add_messages

# from backend.constants import WEAVIATE_DOCS_INDEX_NAME # No longer used
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


OPENAI_MODEL_KEY = "openai_gpt_3_5_turbo"
ANTHROPIC_MODEL_KEY = "anthropic_claude_3_haiku"
FIREWORKS_MIXTRAL_MODEL_KEY = "fireworks_mixtral"
GOOGLE_MODEL_KEY = "google_gemini_pro"
COHERE_MODEL_KEY = "cohere_command"
GROQ_LLAMA_3_MODEL_KEY = "groq_llama_3"
OLLAMA_MODEL_KEY = "ollama_chat" # New key for Ollama

# Determine default LLM based on environment variable
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
if LLM_PROVIDER == "ollama":
    DEFAULT_MODEL_PROVIDER_KEY = OLLAMA_MODEL_KEY
elif LLM_PROVIDER == "openai": # Default to OpenAI if not ollama or if var is something else
    DEFAULT_MODEL_PROVIDER_KEY = OPENAI_MODEL_KEY
else: # Fallback for any other unexpected LLM_PROVIDER value
    DEFAULT_MODEL_PROVIDER_KEY = OPENAI_MODEL_KEY
    print(f"Warning: Invalid LLM_PROVIDER '{LLM_PROVIDER}'. Defaulting to OpenAI.")


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
    messages: Annotated[list[BaseMessage], add_messages]
    # for convenience in evaluations
    answer: str


gpt_3_5 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, streaming=True)
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
ollama_llm = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    model=os.getenv("OLLAMA_LLM_MODEL", "llama2"), # Default model for Ollama LLM
    temperature=0,
)
llm = gpt_3_5.configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="model_name"),
    default_key=DEFAULT_MODEL_PROVIDER_KEY, # Using the env-driven default
    **{
        OPENAI_MODEL_KEY: gpt_3_5, # Ensure OpenAI is explicitly in the map if it can be a default
        ANTHROPIC_MODEL_KEY: claude_3_haiku,
        FIREWORKS_MIXTRAL_MODEL_KEY: fireworks_mixtral,
        GOOGLE_MODEL_KEY: gemini_pro,
        COHERE_MODEL_KEY: cohere_command,
        GROQ_LLAMA_3_MODEL_KEY: groq_llama3,
        OLLAMA_MODEL_KEY: ollama_llm, # Added Ollama
    },
).with_fallbacks(
    [
        gpt_3_5, # Ensure all models in the map are also in fallbacks if desired
        claude_3_haiku,
        fireworks_mixtral,
        gemini_pro,
        cohere_command,
        groq_llama3,
        ollama_llm, # Added Ollama to fallbacks
    ]
)


def get_retriever() -> BaseRetriever:
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "knowledge_base")

    # Ensure Milvus connection is established
    # The alias "default" is used by Langchain Milvus client by default.
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, alias="default")

    vector_store = Milvus(
        embedding_function=get_embeddings_model(),
        collection_name=MILVUS_COLLECTION_NAME,
        connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT, "alias": "default"},
        # Ensure these field names match what ingest.py is using/setting up
        # primary_field="id", # auto_id=True in ingest.py handles primary key, so not needed here.
    # For Milvus, text_field and vector_field are often set during collection creation or by default.
    # If your Milvus collection uses specific names, ensure they are reflected here or in the client.
    # The Milvus client for Langchain typically expects 'text' and 'vector' or similar defaults.
    # If auto_id=True was used in ingest, Milvus manages primary keys.
    # If a specific text field name was used (e.g., 'page_content'), it should be consistent.
    # Default behavior of Milvus client in Langchain is usually sufficient if collection schema is standard.
    text_field="text", # As per previous setup in ingest.py
    vector_field="embedding" # As per previous setup in ingest.py
    )
    return vector_store.as_retriever(search_kwargs=dict(k=6))


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def retrieve_documents(state: AgentState) -> AgentState:
    retriever = get_retriever()
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


def synthesize_response(
    state: AgentState, model: LanguageModelLike, prompt_template: str
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
    return {"messages": [synthesized_response], "answer": synthesized_response.content}


def synthesize_response_default(state: AgentState) -> AgentState:
    return synthesize_response(state, llm, RESPONSE_TEMPLATE)


def synthesize_response_cohere(state: AgentState) -> AgentState:
    model = llm.bind(documents=state["documents"])
    return synthesize_response(state, model, COHERE_RESPONSE_TEMPLATE)


def route_to_response_synthesizer(
    state: AgentState, config: RunnableConfig
) -> Literal["response_synthesizer", "response_synthesizer_cohere"]:
    model_name = config.get("configurable", {}).get("model_name", DEFAULT_MODEL_PROVIDER_KEY)
    if model_name == COHERE_MODEL_KEY:
        return "response_synthesizer_cohere"
    else:
        return "response_synthesizer"


workflow = StateGraph(AgentState)

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
