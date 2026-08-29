"""Regression tests for guardrails scope classification context."""

import asyncio

from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from src.middleware.guardrails_middleware import GuardrailsMiddleware


class FakeClassifier:
    """Return a configured guardrails decision and retain the prompt."""

    def __init__(self, decision):
        self.decision = decision
        self.prompts = []

    def with_structured_output(self, schema):  # noqa: ARG002
        return self

    async def ainvoke(self, prompt, config=None):  # noqa: ARG002
        self.prompts.append(prompt)
        return self.decision


class FakeRejectionModel:
    """Return a minimal rejection response for blocked requests."""

    async def ainvoke(self, prompt):  # noqa: ARG002
        return HumanMessage(content="I only cover LangChain documentation topics.")


def _middleware(decision):
    classifier = FakeClassifier(decision)
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.classifier_llms = [("test", classifier)]
    middleware.block_off_topic = True
    middleware.llm = FakeRejectionModel()
    middleware._add_to_dataset = _noop_add_to_dataset
    return middleware, classifier


async def _noop_add_to_dataset(*args, **kwargs):  # noqa: ARG001
    return None


def test_page_context_allows_non_english_symbol_parameter_question(monkeypatch):
    middleware, classifier = _middleware(
        {
            "decision": "ALLOWED",
            "explanation": "Page context identifies LangChain docs.",
        }
    )
    monkeypatch.setattr("src.middleware.guardrails_middleware.random.random", lambda: 1)
    state = {
        "messages": [
            HumanMessage(
                content=(
                    "Project: langchain\n"
                    "Package: langchain-core\n"
                    "Symbol: similarity_search\n"
                    "Page URL: https://reference.langchain.com/python/\n\n"
                    "como deixo o k sem limite?"
                )
            )
        ]
    }

    result = asyncio.run(middleware.abefore_agent(state, Runtime(context=None)))

    assert result is None
    classifier_prompt = classifier.prompts[0][1].content
    assert "Project: langchain" in classifier_prompt
    assert "Package: langchain-core" in classifier_prompt
    assert "Symbol: similarity_search" in classifier_prompt
    assert "como deixo o k sem limite?" in classifier_prompt


def test_prior_langchain_question_allows_structured_package_summary(monkeypatch):
    middleware, classifier = _middleware(
        {"decision": "ALLOWED", "explanation": "Prior question establishes scope."}
    )
    monkeypatch.setattr("src.middleware.guardrails_middleware.random.random", lambda: 1)
    state = {
        "messages": [
            HumanMessage(content="what is any from langchain import any"),
            HumanMessage(
                content="Prepare a tree diagram of language core and its branches/packages"
            ),
        ]
    }

    result = asyncio.run(middleware.abefore_agent(state, Runtime(context=None)))

    assert result is None
    classifier_prompt = classifier.prompts[0][1].content
    assert "Previous questions in this conversation:" in classifier_prompt
    assert "what is any from langchain import any" in classifier_prompt


def test_off_topic_requests_are_blocked(monkeypatch):
    blocked_requests = [
        "Review this product requirements document.",
        "What are the latest trends in the general AI industry?",
        "Recommend books about Apache Spark.",
    ]
    monkeypatch.setattr("src.middleware.guardrails_middleware.random.random", lambda: 1)

    for request in blocked_requests:
        middleware, _ = _middleware(
            {
                "decision": "BLOCKED",
                "explanation": "Outside LangChain documentation scope.",
            }
        )
        result = asyncio.run(
            middleware.abefore_agent(
                {"messages": [HumanMessage(content=request)]},
                Runtime(context=None),
            )
        )

        assert result["off_topic_query"] is True
        assert result["jump_to"] == "end"
