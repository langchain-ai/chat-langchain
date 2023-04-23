"""Main entrypoint for the app."""
import datetime
import os
import random
import string
import sys
import time
from typing import List, Optional

import pinecone
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, VectorStore
from loguru import logger
from sqlmodel import Session, SQLModel, create_engine

from app.callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from app.models import config
from app.models.schemas import ChatLog, ChatResponse, DataSource
from app.query_data import get_chain
from app.routers import static_router

app = FastAPI()

logger.remove()
logger.add(sys.stderr, level="DEBUG")

settings = config.get_settings()
templates = Jinja2Templates(directory="app/templates")
vectorstore: Optional[VectorStore] = None

engine = create_engine(settings.docchat_database_url, echo=True)
SQLModel.metadata.create_all(engine)

app.mount("/assets", StaticFiles(directory="app/assets"), name="assets")

app.include_router(static_router.router)


@app.on_event("startup")
async def startup_event():
    load_dotenv()
    logger.add(sys.stderr, enqueue=True)
    logger.info("init pinecone vectorstore")
    logger.debug("debug pincone init")
    # initialize pinecone
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.environ.get("PINECONE_ENV"),
    )


@app.middleware("http")
async def log_requests(request: Request, call_next):
    idem = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} start request path={request.url.path}")
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    formatted_process_time = "{0:.2f}".format(process_time)
    logger.info(
        f"rid={idem} completed_in={formatted_process_time}ms "
        f"status_code={response.status_code}"
    )

    return response


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    chat_trace = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
    index = pinecone.Index(os.environ.get("PINECONE_INDEX"))
    logger.debug(f"Pinecone index: {index}")
    embeddings = OpenAIEmbeddings()
    vectorstore = Pinecone(index, embeddings.embed_query, "text")
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())

            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            answer = result["answer"]
            chat_history.append((question, answer))

            log_chat(question, answer, chat_trace)

            logger.debug("Source documents...")
            source_docs = result["source_documents"]
            data_sources = format_data_sources(source_docs)

            end_resp = ChatResponse(
                sender="bot", message="", type="end", sources=data_sources
            )
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logger.info("websocket disconnect")
            break
        except Exception as e:
            logger.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


def format_data_sources(source_docs) -> List[DataSource]:
    logger.debug("formatting data sources")
    data_sources = []
    for source_doc in source_docs:
        page_content = source_doc.page_content
        meta_data = source_doc.metadata
        data_sources.append(
            DataSource(
                page_content=page_content,
                meta_data=meta_data,
            ).dict()
        )
    return data_sources


def log_chat(question: str, answer: str, chat_trace: str):
    logger.info("logging chat...")
    logger.info(f"question: {question}")
    logger.info(f"answer: {answer}")
    logger.info(f"chat_trace: {chat_trace}")
    now = datetime.datetime.now()
    try:
        with Session(engine) as session:
            session.add(
                ChatLog(
                    answer=answer,
                    question=question,
                    chat_trace=chat_trace,
                    time_stamp=now,
                )
            )
            session.commit()
        logger.info("Chat logged.")
    except Exception as e:
        logger.exception(f"An exception occurred while logging chat: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
