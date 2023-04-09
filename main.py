"""Main entrypoint for the app."""
import logging
import os
from typing import List, Optional

import pinecone
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone, VectorStore
from loguru import logger

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from schemas import ChatResponse, DataSource

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    load_dotenv()
    logging.info("init pinecone vectorstore")
    # initialize pinecone
    pinecone.init(
        api_key=os.environ.get("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.environ.get("PINECONE_ENV"),
    )


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
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
            logger.debug(f"answer: {answer}")
            chat_history.append((question, answer))

            logger.debug("Source documents:")
            source_docs = result["source_documents"]
            data_sources = format_data_sources(source_docs)

            logger.debug(data_sources)
            end_resp = ChatResponse(
                sender="bot", message="", type="end", sources=data_sources
            )
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket disconnect")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="Sorry, something went wrong. Try again.",
                type="error",
            )
            await websocket.send_json(resp.dict())


def format_data_sources(source_docs) -> List[DataSource]:
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
