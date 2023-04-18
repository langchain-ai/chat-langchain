"""Main entrypoint for the app."""
import logging
import fnmatch
import os
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from ingest import ingest_docs
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None

@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    if not Path("vectorstore.pkl").exists():
        raise ValueError("vectorstore.pkl does not exist, please run ingest.py first")
    with open("vectorstore.pkl", "rb") as f:
        global vectorstore
        vectorstore = pickle.load(f)

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    question_handler = QuestionGenCallbackHandler(websocket)
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)

    questions = [
    "What is the main idea of this paper/document?",
    "What is the problem or issue addressed in this paper/document?",
    "What methodology or approach is used to address the research problem?",
    "Is the dataset used in this paper/document publicly available? If so, where can it be accessed?",
    "Is the code used in this paper/document publicly available? If so, where can it be accessed?",
    "What are the main findings or results presented in this paper/document?"
    ]

    for file in os.listdir('./docs'):
        if fnmatch.fnmatch(file, '*.pdf'):
            print(file)
            ingest_docs(file)
            for question in questions:
                print(question)
                try:
                    # Receive and send back the client message
                    #question = await websocket.receive_text() #this is what I replace with my preconfigured prompts
                    resp = ChatResponse(sender="you", message=question, type="stream")
                    await websocket.send_json(resp.dict())

                    # Construct a response
                    start_resp = ChatResponse(sender="bot", message="", type="start")
                    await websocket.send_json(start_resp.dict())

                    result = await qa_chain.acall(
                        {"question": question, "chat_history": chat_history}
                    )
                    chat_history.append((question, result["answer"]))

                    end_resp = ChatResponse(sender="bot", message="", type="end")
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

if __name__ == "__main__":
    import uvicorn
    # ingest_docs("1.pdf")
    uvicorn.run(app, host="0.0.0.0", port=9000)
