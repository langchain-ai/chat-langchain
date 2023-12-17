import os
from operator import itemgetter
from typing import Dict, List, Optional, Sequence, Union

import langsmith
import weaviate
import openai
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.azure_openai import AzureOpenAIEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.voyageai import VoyageEmbeddings
from langchain.prompts import (ChatPromptTemplate, MessagesPlaceholder,
                               PromptTemplate)
from langchain.schema import Document
from langchain.schema.embeddings import Embeddings
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.retriever import BaseRetriever
from langchain.schema.runnable import (Runnable, RunnableBranch,
                                       RunnableLambda, RunnableMap)
from langchain.vectorstores import Weaviate
from pydantic import BaseModel

#openai.api_type = "azure"

class CrawlerRequest(BaseModel):
    document_id: str

RESPONSE_TEMPLATE = """\
You are an expert Volksbank assistant, tasked with answering any question \
about Volksbank internal knowladge that your colleagues might have. \
You always answer in the same language as the question you get asked.

Generate a comprehensive and informative answer of 80 words or less for the \
given question based solely on the provided search results (URL and content). You must \
only use information from the provided search results. Use an unbiased and \
journalistic tone. Combine search results together into a coherent answer. Do not \
repeat text. If different results refer to different entities within the same name, write separate \
answers for each entity.

You should answer in short and precise sentences.

If there is nothing in the context relevant to the question at hand, just say "Ich \
habe leider keine Informationen darüber." Don't try to make up an answer.

Anything between the following `context`  html blocks is retrieved from a knowledge \
data base, not part of the conversation with the user. 

<context>
    {context} 
<context/>

REMEMBER: If there is no relevant information within the context, just say "Ich \
habe leider keine Informationen darüber." Don't try to make up an answer. \
Anything between the preceding 'context' html blocks is retrieved from a knowledge bank, \ 
not part of the conversation with the user.\
"""

REPHRASE_TEMPLATE = """\
Given the following conversation and a follow up question, rephrase the follow up \
question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone Question:"""


WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]


class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[List[Dict[str, str]]]

"""
# Use these for OpenAI Embeddings
def get_embeddings_model() -> Embeddings:
    if os.environ.get("VOYAGE_API_KEY") and os.environ.get("VOYAGE_AI_MODEL"):
        return VoyageEmbeddings(model=os.environ["VOYAGE_AI_MODEL"])
    return OpenAIEmbeddings(chunk_size=200)
"""

def get_embeddings_model() -> Embeddings:
    return AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002",openai_api_version="2023-05-15")

def get_retriever() -> BaseRetriever:
    weaviate_client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
    )
    weaviate_client = Weaviate(
        client=weaviate_client,
        index_name=os.environ["WEAVIATE_CLASS"],
        text_key="text",
        embedding=get_embeddings_model(),
        by_text=False,
        attributes=["source", "title", "score"],
    )
    return weaviate_client.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={'score_threshold': 0.7}
            )


def create_retriever_chain(
    llm: BaseLanguageModel, retriever: BaseRetriever
) -> Runnable:
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (
        CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    ).with_config(
        run_name="CondenseQuestion",
    )
    conversation_chain = condense_question_chain | retriever
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
                run_name="HasChatHistoryCheck"
            ),
            conversation_chain.with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(
                run_name="Itemgetter:question"
            )
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = f"<doc id='{i}'>{doc.page_content}</doc>"
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: ChatRequest):
    chat_history = request["chat_history"] or []
    converted_chat_history = []
    for message in chat_history:
        if message.get("human") is not None:
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai") is not None:
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_chain(
    llm: BaseLanguageModel,
    retriever: BaseRetriever,
) -> Runnable:
    retriever_chain = create_retriever_chain(
        llm,
        retriever,
    ).with_config(run_name="FindDocs")
    _context = RunnableMap(
        {
            "context": retriever_chain | format_docs,
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

    response_synthesizer = (prompt | llm | StrOutputParser()).with_config(
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


def create_answer_chain() -> Runnable:
    """
    # Use this for OpenAI API
    llm = ChatOpenAI(
        model="gpt-4-1106-preview",
        streaming=True,
        temperature=0,
    )
    """
    llm = AzureChatOpenAI(
        azure_deployment="gpt-4",
        openai_api_version="2023-05-15",
        streaming=True,
        temperature=0,
    )
    retriever = get_retriever()
    answer_chain = create_chain(
        llm,
        retriever,
    )
    return answer_chain
