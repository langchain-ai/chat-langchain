"""Main entrypoint for the app."""
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from typing import Any, Optional, Generator
import os
import logging

import langchain
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import VectorStore, Weaviate
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.tracers.run_collector import RunCollectorCallbackHandler
from langchain.schema.runnable import RunnableConfig, RunnableMap, RunnablePassthrough
from langchain.schema import format_document
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA

from operator import itemgetter
import weaviate
import asyncio

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None
run_collector = RunCollectorCallbackHandler()
runnable_config = RunnableConfig(callbacks=[run_collector])
run_id = None
from langsmith import Client
client = Client()

@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(request: Request):
    global run_id, access_counter
    run_id = None
    access_counter = 0
    run_collector.traced_runs = []
    
    data = await request.json()
    question = data.get('message')
    model_type = data.get('model')
    chat_history = data.get('history', [])
    
    WEAVIATE_URL=os.environ["WEAVIATE_URL"]
    WEAVIATE_API_KEY=os.environ["WEAVIATE_API_KEY"]

    embeddings = OpenAIEmbeddings()
    client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY))
    print(client.query.aggregate("LangChain_idx").with_meta_count().do())
    weaviate_client = Weaviate(client=client, index_name="LangChain_idx", text_key="text", embedding=embeddings, by_text=False, attributes=["source"])
    
    async def stream():
        result = ''
        try:
            model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo") if model_type == "openai" else ChatAnthropic()

            print("Recieved question: ", question)
            
            _template = """You are an expert programmer, tasked to answer any question about Langchain. Be as helpful as possible. 
            
            Here you have multiple separate sources about Langchain.{context} 
                            
            Conversation History:               
            {history}
            
            Answer the user's question to the best of your ability: {question}
            Helpful Answer:"""

            prompt = PromptTemplate(input_variables=["history", "context", "question"], template=_template)
            memory = ConversationBufferMemory(input_key="question", memory_key="history")
            
            for message in chat_history:
                memory.save_context({'question': message['question']}, {'result': message['result']})
                
            qa_chain = RetrievalQA.from_chain_type(
                model,
                chain_type="stuff",
                retriever=weaviate_client.as_retriever(search_kwargs=dict(k=10)),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt, "memory": memory},
            )
            
            for s in qa_chain.stream(question, config=runnable_config):
                result += s["result"]
                yield s["result"]
                await asyncio.sleep(0)

        except Exception as e:
            logging.error(e)
            yield  "Sorry, something went wrong. Try again." + "\n"

    return StreamingResponse(stream())

access_counter = 0

@app.post("/feedback")
async def send_feedback(request: Request):
    global run_id, access_counter
    data = await request.json()
    score = data.get('score')
    if not run_id:  
        run = run_collector.traced_runs[0]
        run_id = run.id
        access_counter += 1
        if access_counter >= 2:
            run_collector.traced_runs = []
            access_counter = 0
    client.create_feedback(run_id, "user_score", score=score)
    return {"result": "posted feedback successfully", "code": 200}
    
@app.post("/get_trace")
async def get_trace(request: Request):
    global run_id, access_counter
    if run_id == None:
        run = run_collector.traced_runs[0]
        run_id = run.id
        access_counter += 1
        if access_counter >= 2:
            run_collector.traced_runs = []
            access_counter = 0
    url = client.share_run(run_id)
    return url
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
