import sys


class App:
    def __init__(self, scope):
        assert scope["type"] == "http"
        self.scope = scope

    async def __call__(self, receive, send):
        await send(
            {"type": "http.response.start", "status": 200, "headers": [[b"content-type", b"text/plain"]],}
        )
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        message = f"Hello world! From Uvicorn with Gunicorn. Using Python {version}".encode("utf-8")
        await send({"type": "http.response.body", "body": message})


app = App
