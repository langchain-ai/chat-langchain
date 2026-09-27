from src.middleware.response_language_middleware import validate_response_language


def test_rejects_javascript_api_spelling_in_python_fence():
    answer = """```python\nbuilder.addEdge(\"start\", \"end\")\n```"""

    assert not validate_response_language(answer, "python")


def test_rejects_javascript_citation_for_python_answer():
    answer = "See the example in [the docs](https://docs.langchain.com/oss/javascript/langgraph/overview)."

    assert not validate_response_language(answer, "python")


def test_accepts_javascript_answer_with_javascript_citation():
    answer = """```javascript\nconst graph = new StateGraph({});\n```\n\n[Docs](https://docs.langchain.com/oss/javascript/langgraph/overview)"""

    assert validate_response_language(answer, "javascript")
