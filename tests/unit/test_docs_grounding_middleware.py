from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.middleware.docs_grounding_middleware import (
    DocsGroundingMiddleware,
    _draft_targets,
    _invalid_targets,
)


def _request(*tool_messages: ToolMessage) -> ModelRequest:
    return ModelRequest(
        model=None,
        messages=[HumanMessage(content="How should I import this?"), *tool_messages],
    )


def _response(code: str) -> ModelResponse:
    return ModelResponse(result=[AIMessage(content=f"```python\n{code}\n```")])


def test_extracts_import_and_install_targets_from_python_fences():
    targets = _draft_targets(
        _response(
            "from langchain_core.vectorstores import FAISS\n"
            "import langchain_core\n"
            "pip install langchain-core\n"
            "uv add langchain-core"
        )
    )

    assert [(target.kind, target.path, target.symbol) for target in targets] == [
        ("from", "langchain_core.vectorstores", "FAISS"),
        ("import", "langchain_core", None),
        ("install", "langchain-core", None),
        ("install", "langchain-core", None),
    ]


def test_deprecation_warning_does_not_support_warning_example():
    response = _response("from langchain_community.vectorstores import FAISS")
    docs = [
        "The langchain_community package is no longer maintained. "
        "Use langchain_core.vectorstores instead."
    ]

    invalid = _invalid_targets(response, docs)

    assert len(invalid) == 1


def test_check_links_results_are_not_used_as_documentation():
    response = _response("from langchain_core.vectorstores import FAISS")
    request = _request(
        ToolMessage(
            name="check_links",
            content="from langchain_core.vectorstores import FAISS",
            tool_call_id="1",
        )
    )

    invalid = DocsGroundingMiddleware()._validate(request, response)

    assert len(invalid) == 1


def test_retries_once_with_documented_replacement():
    middleware = DocsGroundingMiddleware()
    request = _request(
        ToolMessage(
            name="query_docs_filesystem_docs_by_lang_chain",
            content=(
                "The langchain_community package is no longer maintained. "
                "Use langchain_core.vectorstores instead.\n"
                "from langchain_core.vectorstores import FAISS"
            ),
            tool_call_id="1",
        )
    )
    responses = iter(
        [
            _response("from langchain_community.vectorstores import FAISS"),
            _response("from langchain_core.vectorstores import FAISS"),
        ]
    )
    seen_requests = []

    def handler(retry_request):
        seen_requests.append(retry_request)
        return next(responses)

    result = middleware.wrap_model_call(request, handler)

    assert result.result[0].content == (
        "```python\nfrom langchain_core.vectorstores import FAISS\n```"
    )
    assert len(seen_requests) == 2
    assert "no documented replacement" in seen_requests[1].system_prompt.lower()
