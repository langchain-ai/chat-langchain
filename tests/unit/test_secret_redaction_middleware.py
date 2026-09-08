import asyncio

from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage

from src.middleware.secret_redaction_middleware import SecretRedactionMiddleware


def test_redacts_secrets_in_string_content():
    middleware = SecretRedactionMiddleware()
    message = AIMessage(content="client = 'sk-1234567890abcdef'")

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[message])

    request = ModelRequest(model=object(), messages=[])
    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert response.result[0].content == "client = 'YOUR_API_KEY_HERE'"
    assert "sk-1234567890abcdef" not in response.result[0].content


def test_redacts_text_blocks_and_preserves_non_text_content():
    middleware = SecretRedactionMiddleware()
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    message = AIMessage(
        content=[
            {"type": "text", "text": "key = 'ghp_12345678901234567890'"},
            image,
        ]
    )

    async def handler(request: ModelRequest) -> ModelResponse:
        return ModelResponse(result=[message])

    request = ModelRequest(model=object(), messages=[])
    response = asyncio.run(middleware.awrap_model_call(request, handler))

    assert response.result[0].content == [
        {"type": "text", "text": "key = 'YOUR_API_KEY_HERE'"},
        image,
    ]
    assert "ghp_12345678901234567890" not in str(response.result[0].content)
