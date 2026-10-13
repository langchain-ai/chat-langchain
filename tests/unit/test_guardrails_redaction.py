"""Tests for guardrail redaction and a synthetic request through the agent."""

import asyncio
import json
import os
from unittest.mock import AsyncMock, Mock

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tracers.langchain import LangChainTracer
from langgraph.runtime import Runtime
from langsmith import Client

os.environ["USE_LOCAL_PROMPTS"] = "1"

from src.middleware import guardrails_middleware as guardrails_module
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from src.utils.secret_redaction import REDACTED_SECRET, create_secret_anonymizer


class RecordingClassifier:
    def __init__(self, decision="ALLOWED"):
        self.decision = decision
        self.prompts = []

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return {"decision": self.decision, "explanation": "Synthetic classification."}


def _middleware(classifier):
    middleware = GuardrailsMiddleware.__new__(GuardrailsMiddleware)
    middleware.classifier_llms = [("synthetic", classifier)]
    middleware.block_off_topic = False
    return middleware


def _dataset_client(monkeypatch):
    client = AsyncMock()
    client.__aenter__.return_value = client
    monkeypatch.setattr(guardrails_module.ls, "AsyncClient", lambda: client)
    monkeypatch.setattr(guardrails_module, "_dataset_id_cache", "synthetic-dataset")
    return client


@pytest.mark.parametrize(
    "secret, credential",
    [
        ("lsv2_pt_" + "a" * 32, "lsv2_pt_" + "a" * 32),
        ('aws_secret_access_key: "synthetic-secret"', "synthetic-secret"),
        ("postgres://user:synthetic-password@host/db", "synthetic-password"),
    ],
)
def test_classifier_redacts_raw_content_and_legacy_history(secret, credential):
    image = {"type": "image_url", "image_url": {"url": "https://example.invalid/image"}}
    classifier = RecordingClassifier()
    middleware = _middleware(classifier)
    content = [secret, {"type": "text", "text": f"Please debug {secret}"}, image]
    asyncio.run(
        middleware._classify_query(
            [HumanMessage(content=content)],
            [{"query": secret, "decision": "ALLOWED"}],
        )
    )

    serialized = json.dumps([message.content for message in classifier.prompts[0]])
    assert credential not in serialized
    assert REDACTED_SECRET in serialized
    assert classifier.prompts[0][-1].content[-1] == image
    assert content[0] == secret
    rejection = middleware._build_rejection_content(content)
    assert credential not in json.dumps(rejection)
    assert rejection[-2] == image


@pytest.mark.parametrize(
    "secret, expected",
    [
        ("sk-proj-" + "b" * 32, REDACTED_SECRET),
        (
            'aws_secret_access_key: "synthetic-secret"',
            f'aws_secret_access_key: "{REDACTED_SECRET}"',
        ),
        (
            "postgres://user:synthetic-password@host/db",
            f"postgres://user:{REDACTED_SECRET}@host/db",
        ),
    ],
)
def test_dataset_redacts_query_and_explanation_independently(
    monkeypatch, secret, expected
):
    client = _dataset_client(monkeypatch)
    middleware = _middleware(RecordingClassifier())
    asyncio.run(middleware._add_to_dataset(secret, "BLOCKED", secret, secret))

    sample = client.create_example.call_args.kwargs
    assert sample["inputs"]["query"] == expected
    assert sample["outputs"]["explanation"] == expected
    assert secret not in json.dumps(sample)


@pytest.mark.parametrize(
    "secret, expected",
    [
        ("lsv2_pt_" + "c" * 32, REDACTED_SECRET),
        (
            'aws_secret_access_key: "synthetic-secret"',
            f'aws_secret_access_key: "{REDACTED_SECRET}"',
        ),
        (
            "postgres://user:synthetic-password@host/db",
            f"postgres://user:{REDACTED_SECRET}@host/db",
        ),
    ],
)
def test_guardrail_sampling_redacts_without_ingress(monkeypatch, secret, expected):
    client = _dataset_client(monkeypatch)
    classifier = RecordingClassifier("BLOCKED")
    middleware = _middleware(classifier)

    async def run():
        update = await middleware.abefore_agent(
            {"messages": [HumanMessage(content=secret)]}, Runtime(context=None)
        )
        await asyncio.sleep(0)
        return update

    update = asyncio.run(run())
    assert update["guardrail_history"][0]["query"] == expected
    assert client.create_example.call_args.kwargs["inputs"]["query"] == expected
    assert secret not in str(classifier.prompts[0])


def test_synthetic_request_redacts_model_classifier_dataset_and_trace(monkeypatch):
    secrets = [
        "lsv2_pt_" + "d" * 32,
        "synthetic/AwsSecret+value=",
        "synthetic-password",
    ]
    content = (
        f"LANGSMITH_API_KEY={secrets[0]}\n"
        f'aws_secret_access_key: "{secrets[1]}"\n'
        f"postgres://user:{secrets[2]}@host:5432/db"
    )
    dataset = _dataset_client(monkeypatch)
    monkeypatch.setattr(guardrails_module, "ALLOWED_SAMPLE_RATE", 1)
    classifier = RecordingClassifier()
    model = FakeMessagesListChatModel(responses=[AIMessage(content=REDACTED_SECRET)])
    model_inputs = []
    original_generate = FakeMessagesListChatModel._generate

    def record_generate(self, messages, *args, **kwargs):
        model_inputs.extend(message.content for message in messages)
        return original_generate(self, messages, *args, **kwargs)

    monkeypatch.setattr(FakeMessagesListChatModel, "_generate", record_generate)
    session = Mock()
    session.request.return_value.status_code = 200
    client = Client(
        api_url="https://langsmith.invalid",
        api_key="test",
        session=session,
        info={"version": "0.0.0"},
        auto_batch_tracing=False,
        anonymizer=create_secret_anonymizer(),
    )
    tracer = LangChainTracer(client=client)
    graph = create_agent(
        model=model,
        middleware=[IngressGuardsMiddleware(), _middleware(classifier)],
    )

    async def run():
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=content, id="h1")]},
            config={"callbacks": [tracer]},
        )
        await asyncio.sleep(0)
        return result

    result = asyncio.run(run())
    tracer.wait_for_futures()

    assert result["messages"][0].id == "h1"
    assert all(secret not in str(result["messages"]) for secret in secrets)
    assert model_inputs and REDACTED_SECRET in str(model_inputs)
    assert all(secret not in str(model_inputs) for secret in secrets)
    assert all(secret not in str(classifier.prompts) for secret in secrets)
    assert REDACTED_SECRET in str(classifier.prompts)
    sample = dataset.create_example.call_args.kwargs
    assert all(secret not in json.dumps(sample) for secret in secrets)
    assert REDACTED_SECRET in json.dumps(sample)
    writes = [
        json.loads(request.kwargs["data"]) for request in session.request.call_args_list
    ]
    assert writes
    assert all(secret not in json.dumps(writes) for secret in secrets)
    root_writes = [write for write in writes if write.get("name") == "LangGraph"]
    assert root_writes
    assert REDACTED_SECRET in json.dumps(root_writes)
    assert any(
        "outputs" in write and REDACTED_SECRET in json.dumps(write["outputs"])
        for write in writes
    )
    client.close()
