# chat-your-data app
chat-your-data web application. A simple web application that allows you to chat with your data.

Built with [LangChain](https://github.com/hwchase17/langchain/) and [FastAPI](https://fastapi.tiangolo.com/).

App leverages LangChain's streaming support and async API to update the page in real time and support multiple users.

To run:
1. Install dependencies: `pip install -r requirements.txt`
2. Run `python ingest.py` to ingest data into the vectorstore (only needs to be done once).
2. Run the app: `make start`
3. To enable tracing, make sure `langchain-server` is running locally and pass `tracing=True` to `get_chain` in `main.py`.

![Chat_Your_Data.gif](assets%2Fimages%2FChat_Your_Data.gif)