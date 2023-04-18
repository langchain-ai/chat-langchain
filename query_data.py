from langchain.callbacks.base import AsyncCallbackManager
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.chat_vector_db.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# system_template="""Use the following pieces of context to answer the users question. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# ----------------
# {context}"""

system_template="""You are an AI assistant providing helpful advice. You are given the following extracted parts of a long document and a question. Provide a conversational answer based on the context provided.
You should only provide hyperlinks that reference the context below. Do NOT make up hyperlinks.
If you can't find the answer in the context below, just say "Hmm, I'm not sure." Don't try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

Question: {question}
=========
{context}
=========
Answer in Markdown:"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")
]
prompt = ChatPromptTemplate.from_messages(messages)

def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler
) -> ConversationalRetrievalChain:
    manager = AsyncCallbackManager([])
    question_manager = AsyncCallbackManager([question_handler])
    stream_manager = AsyncCallbackManager([stream_handler])

    question_gen_llm = ChatOpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
        openai_api_key="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"
    )
    streaming_llm = ChatOpenAI(
        streaming=True,
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
        openai_api_key="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"
    )

    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT, callback_manager=manager
    )
    
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=prompt, callback_manager=manager
    )

    chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        callback_manager=manager
    )
    return chain