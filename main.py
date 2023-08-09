"""Main entrypoint for the app."""
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from typing import Any, Optional
import os
import logging

import langchain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores import VectorStore, Weaviate

import weaviate
from schemas import ChatResponse
import asyncio

# langchain.debug = True

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    WEAVIATE_URL=os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY=os.environ["WEAVIATE_API_KEY"]

    await websocket.accept()
    
    # if model_type == "anthropic":
    #     model = ChatAnthropic()
    # else:
    model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

    chat_history = []
    embeddings = OpenAIEmbeddings()
    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))
    print(client.query.aggregate("LangChain_idx").with_meta_count().do())
    weaviate_client = Weaviate(client=client, index_name="LangChain_idx", text_key="text", embedding=embeddings, by_text=False)

    while True:
        try:
            # Receive and send back the client message
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())
            await asyncio.sleep(0)

            print("Recieved question: ", question)

            # Construct a response
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            
            prompt = ChatPromptTemplate.from_template("Using Langchain docs, help me answer a question about {question} with the context provided {context}")
            chain = (
                {"context": weaviate_client.as_retriever(), "question": RunnablePassthrough()} 
                | prompt 
                | model 
            )

            result = ""
            
            for s in chain.stream(question):
                resp = ChatResponse(sender="bot", message=s.content, type="stream")
                await websocket.send_json(resp.dict())
                await asyncio.sleep(0)
                result += s.content
                
            chat_history.append((question, result))

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

    uvicorn.run(app, host="0.0.0.0", port=9000)
