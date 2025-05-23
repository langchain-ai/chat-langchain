from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
import uvicorn # Ensure uvicorn is imported if used in __main__

# Assuming your compiled graph is named 'graph' in backend.graph
from backend.graph import graph 

app = FastAPI(
    title="LangChain Server for Custom Chatbot",
    version="1.0",
    description="Serves the custom chatbot graph.",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Add routes for the graph
# The path "/chat_langchain" should match DEFAULT_LANGSERVE_URL in streamlit_app.py
add_routes(
    app,
    graph,
    path="/chat_langchain", 
    config_keys=["configurable"], # Exposes configurable fields like model_name from the graph
    # enable_feedback_endpoint=True, # Optional: if you want to collect feedback via LangSmith
    # per_req_config_modifier=my_modifier_function, # Optional: for more complex config logic
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
