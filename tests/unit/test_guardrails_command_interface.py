import asyncio

from langchain_core.messages import HumanMessage

from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.prompts.guardrails_prompts import guardrails_system_prompt


class RecordingStructuredModel:
    def __init__(self, result):
        self.result = result
        self.prompts = []

    def with_structured_output(self, schema):  # noqa: ARG002
        return self

    async def ainvoke(self, prompt, config=None):  # noqa: ARG002
        self.prompts.append(prompt)
        return self.result


def _middleware(model):
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.classifier_llms = [("test", model)]
    middleware.block_off_topic = True
    return middleware


def test_user_defined_invocation_is_classified_as_blocked():
    model = RecordingStructuredModel(
        {"decision": "BLOCKED", "explanation": "Unverified interface."}
    )
    middleware = _middleware(model)

    result = asyncio.run(
        middleware._classify_query([HumanMessage(content="LangChain@runtime: model --help")])
    )

    assert result["decision"] == "BLOCKED"
    assert "user-defined" in model.prompts[0][0].content


def test_refusal_marker_is_included_for_reissued_invocation():
    model = RecordingStructuredModel(
        {"decision": "BLOCKED", "explanation": "Previously refused request."}
    )
    middleware = _middleware(model)

    result = asyncio.run(
        middleware._classify_query(
            [HumanMessage(content="LangChain@runtime: model --help")],
            "A previous request was refused by guardrails.",
        )
    )

    assert result["decision"] == "BLOCKED"
    assert "Previously refused in this conversation" in model.prompts[0][1].content


def test_blocked_request_persists_refusal_marker(monkeypatch):
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.block_off_topic = True

    async def classify(messages, marker=None):  # noqa: ARG001
        return {"decision": "BLOCKED", "explanation": "Unverified interface."}

    async def reject(content):  # noqa: ARG001
        return HumanMessage(content="I cannot help with that request.")

    monkeypatch.setattr(middleware, "_classify_query", classify)
    monkeypatch.setattr(middleware, "_generate_rejection_message", reject)
    result = asyncio.run(
        middleware.abefore_agent(
            {"messages": [HumanMessage(content="$Product cmd")]}, None
        )
    )

    assert "guardrails_refused_request" in result


def test_docs_prompt_requires_grounded_format_and_documented_commands():
    prompt = guardrails_system_prompt
    assert "user-defined invocation syntax" in prompt

    from src.prompts.docs_agent_prompt import docs_agent_prompt

    assert "cannot be waived" in docs_agent_prompt
    assert "NEVER invent command syntax" in docs_agent_prompt
