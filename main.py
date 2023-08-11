"""Main entrypoint for the app."""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from typing import Any, Optional, Generator
import os
import logging
import json

import langchain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    question = data.get('message')
    model_type = data.get('model')
    
    WEAVIATE_URL=os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY=os.environ["WEAVIATE_API_KEY"]

    chat_history = []
    embeddings = OpenAIEmbeddings()
    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))
    print(client.query.aggregate("LangChain_idx").with_meta_count().do())
    weaviate_client = Weaviate(client=client, index_name="LangChain_idx", text_key="text", embedding=embeddings, by_text=False)

    async def stream():
        try:
            model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo") if model_type == "openai" else ChatAnthropic()

            print("Recieved question: ", question)
            
            prompt = ChatPromptTemplate.from_template("You are an expert programming prodigy creator of Langchain. Answer this question to the best of your ability {question} with the provided context {context}")
            chain = (
                {"context": weaviate_client.as_retriever(), "question": RunnablePassthrough()} 
                | prompt 
                | model 
            )

            result = ""
            
            for s in chain.stream(question):
                yield formatted_content
                await asyncio.sleep(0)
                print(formatted_content, end="", flush=True)
                
            result += formatted_content
            chat_history.append((question, result))

        except Exception as e:
            logging.error(e)
            yield  "Sorry, something went wrong. Try again." + "\n"

    return StreamingResponse(stream())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

