"""Tests for the thread-pinned post-trained model experiment."""

import asyncio
import itertools
import os
import uuid

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.middleware.model_experiment_middleware import (
    ARM_CONTROL,
    ARM_TREATMENT,
    ModelArmPinMiddleware,
    ModelChain,
    ModelExperiment,
    ModelExperimentFallbackMiddleware,
    assign_arm,
)
from src.middleware.model_fallback_middleware import RepairingModelFallbackMiddleware


def _model(name: str) -> GenericFakeChatModel:
    # A fresh message per call: a reused message keeps its id, and the
    # add_messages reducer would replace the earlier answer instead of appending.
    return GenericFakeChatModel(
        messages=(AIMessage(content=name) for _ in itertools.count())
    )


class _FailingModel(GenericFakeChatModel):
    def _generate(self, *args, **kwargs):
        raise RuntimeError("model down")


def _failing() -> _FailingModel:
    return _FailingModel(messages=iter(()))


def _experiment(
    *,
    percent: int = 100,
    treatment: ModelChain | None = None,
    enabled: bool = True,
) -> ModelExperiment:
    if enabled and treatment is None:
        treatment = ModelChain(_model("treatment"), (_model("treatment-fallback"),))
    return ModelExperiment(
        name="test-experiment",
        control=ModelChain(_model("control"), (_model("control-fallback"),)),
        treatment=treatment if enabled else None,
        treatment_percent=percent,
    )


def _agent(experiment: ModelExperiment, checkpointer: InMemorySaver):
    return create_agent(
        model=_model("unused"),
        middleware=[
            ModelArmPinMiddleware(experiment),
            ModelExperimentFallbackMiddleware(experiment),
        ],
        checkpointer=checkpointer,
    )


def _ask(agent, thread_id: str, text: str = "hi") -> str:
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [HumanMessage(content=text)]}, config)
    return result["messages"][-1].content


def _pinned_arm(agent, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    return agent.get_state(config).values.get("model_arm")


# --- assignment -------------------------------------------------------------


def test_assign_arm_is_deterministic_and_salted():
    thread_id = str(uuid.uuid4())
    assert assign_arm(thread_id, "a", 30) == assign_arm(thread_id, "a", 30)
    ids = [str(uuid.uuid4()) for _ in range(2000)]
    assert [assign_arm(t, "a", 50) for t in ids] != [
        assign_arm(t, "b", 50) for t in ids
    ]


def test_assign_arm_hits_the_requested_share():
    ids = [str(uuid.uuid4()) for _ in range(20_000)]
    share = sum(assign_arm(t, "salt", 30) == ARM_TREATMENT for t in ids) / len(ids)
    assert 0.28 < share < 0.32
    assert all(assign_arm(t, "salt", 0) == ARM_CONTROL for t in ids[:500])
    assert all(assign_arm(t, "salt", 100) == ARM_TREATMENT for t in ids[:500])


def test_treatment_percent_is_bounded():
    with pytest.raises(ValueError):
        _experiment(percent=101)


# --- end to end through create_agent ----------------------------------------


def test_treatment_thread_stays_on_treatment_across_turns_and_config_changes():
    saver = InMemorySaver()
    thread_id = str(uuid.uuid4())
    treatment = ModelChain(_model("treatment"), (_model("treatment-fallback"),))

    first = _agent(_experiment(percent=100, treatment=treatment), saver)
    assert _ask(first, thread_id) == "treatment"
    assert _pinned_arm(first, thread_id) == ARM_TREATMENT

    # Enrollment closed: the pinned thread keeps its arm.
    closed = _agent(_experiment(percent=0, treatment=treatment), saver)
    assert _ask(closed, thread_id, "follow up") == "treatment"
    # New threads go to control.
    assert _ask(closed, str(uuid.uuid4())) == "control"


def test_control_thread_stays_on_control_when_percent_rises():
    saver = InMemorySaver()
    thread_id = str(uuid.uuid4())
    assert _ask(_agent(_experiment(percent=0), saver), thread_id) == "control"
    assert (
        _ask(_agent(_experiment(percent=100), saver), thread_id, "again") == "control"
    )


def test_kill_switch_moves_pinned_treatment_threads_to_control():
    saver = InMemorySaver()
    thread_id = str(uuid.uuid4())
    assert _ask(_agent(_experiment(percent=100), saver), thread_id) == "treatment"
    killed = _agent(_experiment(enabled=False), saver)
    assert _ask(killed, thread_id, "again") == "control"


def test_threads_answered_before_the_experiment_stay_on_control():
    saver = InMemorySaver()
    thread_id = str(uuid.uuid4())
    legacy = create_agent(
        model=_model("unused"),
        middleware=[
            RepairingModelFallbackMiddleware(
                _model("control-fallback"), primary_model=_model("control")
            )
        ],
        checkpointer=saver,
    )
    assert _ask(legacy, thread_id) == "control"
    experiment_agent = _agent(_experiment(percent=100), saver)
    assert _ask(experiment_agent, thread_id, "again") == "control"
    assert _pinned_arm(experiment_agent, thread_id) == ARM_CONTROL


def test_arm_is_hidden_from_run_output_and_ignored_in_input():
    saver = InMemorySaver()
    agent = _agent(_experiment(percent=0), saver)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(
        {"messages": [HumanMessage(content="hi")], "model_arm": ARM_TREATMENT},
        config,
    )
    assert "model_arm" not in result
    assert result["messages"][-1].content == "control"


def test_treatment_failure_falls_back_within_the_treatment_chain():
    saver = InMemorySaver()
    treatment = ModelChain(_failing(), (_model("treatment-fallback"),))
    agent = _agent(_experiment(percent=100, treatment=treatment), saver)
    assert _ask(agent, str(uuid.uuid4())) == "treatment-fallback"


def test_async_run_uses_the_pinned_arm():
    saver = InMemorySaver()
    agent = _agent(_experiment(percent=100), saver)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    async def run():
        return await agent.ainvoke({"messages": [HumanMessage(content="hi")]}, config)

    assert asyncio.run(run())["messages"][-1].content == "treatment"


def test_no_thread_id_runs_control():
    agent = create_agent(
        model=_model("unused"),
        middleware=[
            ModelArmPinMiddleware(_experiment(percent=100)),
            ModelExperimentFallbackMiddleware(_experiment(percent=100)),
        ],
    )
    result = agent.invoke({"messages": [HumanMessage(content="hi")]})
    assert result["messages"][-1].content == "control"


# --- production wiring --------------------------------------------------------


@pytest.fixture
def config_module(monkeypatch):
    for key in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(key, os.environ.get(key) or "test-key")
    from src.agent import config

    return config


def test_pin_middleware_runs_first_and_fallback_is_arm_aware(config_module):
    from agent import docs_agent_middleware

    assert docs_agent_middleware[0] is config_module.model_arm_pin_middleware
    assert isinstance(
        config_module.model_fallback_middleware, ModelExperimentFallbackMiddleware
    )
    assert config_module.model_fallback_middleware in docs_agent_middleware


def test_experiment_is_off_until_a_model_id_is_set(config_module):
    experiment = config_module.model_experiment
    if config_module.FT_MODEL_ID is not None:
        pytest.skip("FT_MODEL_ID is set")
    assert not experiment.enabled
    assert experiment.chain_for(ARM_TREATMENT) is experiment.control
    assert experiment.control.primary.metadata["model_role"] == "primary"
    assert all(
        m.metadata["model_role"] == "fallback" for m in experiment.control.fallbacks
    )


def test_treatment_chain_is_a_labelled_fireworks_lora(config_module, monkeypatch):
    monkeypatch.setattr(
        config_module, "FT_MODEL_ID", "accounts/acme/models/clc-docs-lora"
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    chain = config_module._treatment_chain()

    assert type(chain.primary).__name__ == "ChatFireworks"
    assert chain.primary.model_name == "accounts/acme/models/clc-docs-lora"
    assert chain.primary.metadata["model_arm"] == ARM_TREATMENT
    assert chain.primary.metadata["model_role"] == "primary"
    assert (
        chain.primary.metadata["model_experiment"] == config_module.FT_MODEL_EXPERIMENT
    )
    assert [m.metadata["model_role"] for m in chain.fallbacks] == ["fallback"] * 3
    assert {m.metadata["model_arm"] for m in chain.fallbacks} == {ARM_TREATMENT}
    # Treatment falls back to the control model first.
    assert "gemini" in getattr(chain.fallbacks[0], "model", "")


def test_treatment_chain_requires_a_fireworks_key(config_module, monkeypatch):
    monkeypatch.setattr(config_module, "FT_MODEL_ID", "accounts/acme/models/x")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FIREWORKS_API_KEY"):
        config_module._treatment_chain()


# --- deployment environment ---------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "  "])
def test_unset_model_id_turns_the_experiment_off(config_module, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FT_MODEL_ID", raising=False)
    else:
        monkeypatch.setenv("FT_MODEL_ID", value)
    assert config_module._ft_model_id_from_env() is None


def test_model_id_is_read_from_env(config_module, monkeypatch):
    monkeypatch.setenv("FT_MODEL_ID", " accounts/acme/models/clc-docs-lora ")
    assert config_module._ft_model_id_from_env() == "accounts/acme/models/clc-docs-lora"


@pytest.mark.parametrize(
    "value",
    [
        "clc-docs-lora",
        "accounts/acme/models/clc-docs-lora#accounts/acme/deployments/abc",
        "fireworks:accounts/acme/models/clc-docs-lora",
    ],
)
def test_malformed_model_id_fails_at_startup(config_module, monkeypatch, value):
    monkeypatch.setenv("FT_MODEL_ID", value)
    with pytest.raises(ValueError, match="FT_MODEL_ID"):
        config_module._ft_model_id_from_env()


@pytest.mark.parametrize(
    ("value", "expected"), [(None, 33), ("", 33), ("0", 0), (" 45 ", 45), ("100", 100)]
)
def test_traffic_percent_is_read_from_env(config_module, monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("FT_MODEL_TRAFFIC_PERCENT", raising=False)
    else:
        monkeypatch.setenv("FT_MODEL_TRAFFIC_PERCENT", value)
    assert config_module._ft_traffic_percent_from_env() == expected


@pytest.mark.parametrize("value", ["-1", "101", "30%", "0.3", "thirty"])
def test_bad_traffic_percent_fails_at_startup(config_module, monkeypatch, value):
    monkeypatch.setenv("FT_MODEL_TRAFFIC_PERCENT", value)
    with pytest.raises(ValueError, match="FT_MODEL_TRAFFIC_PERCENT"):
        config_module._ft_traffic_percent_from_env()


# --- root run metadata --------------------------------------------------------


@pytest.fixture(params=["linked", "detached"])
def root_run(monkeypatch, request):
    from types import SimpleNamespace

    from langsmith.run_trees import RunTree

    root = RunTree(name="root", extra={"metadata": {"existing": True}})

    class Client:
        def __init__(self):
            self.feedback = []

        def create_feedback(self, *, run_id, trace_id, key, score=None, value=None):
            # Trace-level feedback must name the root run, or LangSmith
            # treats it as session-level feedback.
            assert run_id == trace_id == root.id
            self.feedback.append((trace_id, key, score if value is None else value))

    root.ls_client = Client()
    child = (
        SimpleNamespace(metadata={}, parent_run=root)
        if request.param == "linked"
        else root.create_child(name="model")
    )
    monkeypatch.setattr(
        "src.middleware.model_experiment_middleware.ls.get_current_run_tree",
        lambda: child,
    )
    detached = request.param == "detached"
    return SimpleNamespace(
        root=root,
        metadata=child.metadata if detached else root.metadata,
        feedback=lambda: dict((k, v) for _, k, v in root.client.feedback),
        detached=detached,
    )


def test_root_run_is_tagged_with_the_arm_on_every_turn(root_run):
    saver = InMemorySaver()
    thread_id = str(uuid.uuid4())
    agent = _agent(_experiment(percent=100), saver)
    assert _ask(agent, thread_id) == "treatment"
    root_run.metadata.clear()
    root_run.root.client.feedback.clear()

    assert _ask(agent, thread_id, "follow up") == "treatment"
    expected = {"model_experiment": "test-experiment", "model_arm": ARM_TREATMENT}
    assert expected.items() <= root_run.metadata.items()
    assert "model_fallback" not in root_run.metadata
    assert root_run.feedback() == {
        **expected,
        "model_arm_treatment": 1,
        "model_arm_base": 0,
    }


def test_control_turns_score_zero_for_treatment(root_run):
    agent = _agent(_experiment(percent=0), InMemorySaver())
    assert _ask(agent, str(uuid.uuid4())) == "control"
    assert root_run.feedback()["model_arm"] == ARM_CONTROL
    assert root_run.feedback()["model_arm_treatment"] == 0


def test_root_run_shows_the_arm_actually_run_after_the_kill_switch(root_run):
    saver = InMemorySaver()
    thread_id = str(uuid.uuid4())
    _ask(_agent(_experiment(percent=100), saver), thread_id)
    root_run.metadata.clear()
    root_run.root.client.feedback.clear()
    assert _ask(_agent(_experiment(enabled=False), saver), thread_id) == "control"
    assert "model_arm" not in root_run.metadata
    assert root_run.feedback() == {}


def test_root_run_is_marked_when_the_turn_falls_back(root_run):
    treatment = ModelChain(_failing(), (_model("treatment-fallback"),))
    agent = _agent(_experiment(percent=100, treatment=treatment), InMemorySaver())
    assert _ask(agent, str(uuid.uuid4())) == "treatment-fallback"
    assert root_run.metadata["model_arm"] == ARM_TREATMENT
    assert root_run.metadata["model_fallback"] is True
    assert root_run.feedback()["model_fallback"] == 1


def test_root_tagging_failure_does_not_fail_the_turn(monkeypatch):
    def boom():
        raise RuntimeError("tracing down")

    monkeypatch.setattr(
        "src.middleware.model_experiment_middleware.ls.get_current_run_tree", boom
    )
    agent = _agent(_experiment(percent=100), InMemorySaver())
    assert _ask(agent, str(uuid.uuid4())) == "treatment"


# --- three-arm split: treatment / base / control -----------------------------


def _three_arm(treatment_percent: int = 33, base_percent: int = 33, base=True):
    return ModelExperiment(
        name="test-experiment",
        control=ModelChain(_model("control"), (_model("control-fallback"),)),
        treatment=ModelChain(_model("treatment"), (_model("treatment-fallback"),)),
        treatment_percent=treatment_percent,
        base=ModelChain(_model("base"), (_model("base-fallback"),)) if base else None,
        base_percent=base_percent,
    )


def test_three_arm_split_is_about_a_third_each():
    from src.middleware.model_experiment_middleware import ARM_BASE

    ids = [str(uuid.uuid4()) for _ in range(6000)]
    arms = [assign_arm(t, "salt", 33, 33) for t in ids]
    for arm in (ARM_TREATMENT, ARM_BASE, ARM_CONTROL):
        assert abs(arms.count(arm) / len(ids) - 1 / 3) < 0.03


def test_base_thread_runs_and_stays_on_the_base_model():
    saver = InMemorySaver()
    thread_id = str(uuid.uuid4())
    agent = _agent(_three_arm(treatment_percent=0, base_percent=100), saver)
    assert _ask(agent, thread_id) == "base"
    assert _ask(agent, thread_id, "follow up") == "base"


def test_without_a_base_chain_its_share_goes_to_control():
    exp = _three_arm(treatment_percent=0, base_percent=100, base=False)
    assert not exp.base_enabled and exp.effective_base_percent == 0
    assert _ask(_agent(exp, InMemorySaver()), str(uuid.uuid4())) == "control"


def test_kill_switch_also_turns_off_the_base_arm():
    from src.middleware.model_experiment_middleware import ARM_BASE

    saver = InMemorySaver()
    thread_id = str(uuid.uuid4())
    assert _ask(_agent(_three_arm(0, 100), saver), thread_id) == "base"
    killed = ModelExperiment(
        name="test-experiment",
        control=ModelChain(_model("control"), (_model("control-fallback"),)),
        treatment=None,
        treatment_percent=0,
        base=ModelChain(_model("base"), ()),
        base_percent=100,
    )
    assert killed.chain_for(ARM_BASE) is killed.control
    assert _ask(_agent(killed, saver), thread_id, "again") == "control"


def test_arm_shares_cannot_exceed_100():
    with pytest.raises(ValueError, match="at most 100"):
        _three_arm(treatment_percent=60, base_percent=50)


def test_base_turns_score_on_model_arm_base(root_run):
    agent = _agent(_three_arm(treatment_percent=0, base_percent=100), InMemorySaver())
    assert _ask(agent, str(uuid.uuid4())) == "base"
    feedback = root_run.feedback()
    assert feedback["model_arm"] == "base"
    assert feedback["model_arm_base"] == 1
    assert feedback["model_arm_treatment"] == 0


def test_base_chain_is_a_labelled_fireworks_model(config_module, monkeypatch):
    from src.middleware.model_experiment_middleware import ARM_BASE

    monkeypatch.setattr(config_module, "FT_MODEL_ID", "accounts/acme/models/lora")
    monkeypatch.setattr(
        config_module, "FT_BASE_MODEL_ID", "accounts/fireworks/models/base-model"
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    chain = config_module._base_chain()
    assert type(chain.primary).__name__ == "ChatFireworks"
    assert chain.primary.model_name == "accounts/fireworks/models/base-model"
    assert chain.primary.metadata["model_arm"] == ARM_BASE
    assert {m.metadata["model_arm"] for m in chain.fallbacks} == {ARM_BASE}
    assert "gemini" in getattr(chain.fallbacks[0], "model", "")

    monkeypatch.setattr(config_module, "FT_MODEL_ID", None)
    assert config_module._base_chain() is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, "high"),
        ("", "high"),
        ("LOW", "low"),
        ("max", "max"),
        ("default", None),
    ],
)
def test_reasoning_effort_is_read_from_env(config_module, monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("FT_REASONING_EFFORT", raising=False)
    else:
        monkeypatch.setenv("FT_REASONING_EFFORT", raw)
    assert config_module._ft_reasoning_effort_from_env() == expected


@pytest.mark.parametrize("raw", ["none", "medium", "3"])
def test_bad_reasoning_effort_fails_at_startup(config_module, monkeypatch, raw):
    monkeypatch.setenv("FT_REASONING_EFFORT", raw)
    with pytest.raises(ValueError, match="FT_REASONING_EFFORT"):
        config_module._ft_reasoning_effort_from_env()


def test_fireworks_arms_send_the_reasoning_effort(config_module, monkeypatch):
    monkeypatch.setattr(config_module, "FT_MODEL_ID", "accounts/acme/models/lora")
    monkeypatch.setattr(
        config_module, "FT_BASE_MODEL_ID", "accounts/fireworks/models/base-model"
    )
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    monkeypatch.setattr(config_module, "FT_REASONING_EFFORT", "high")
    monkeypatch.setattr(config_module, "FT_TREATMENT_REASONING_EFFORT", "low")
    treatment, base = config_module._treatment_chain(), config_module._base_chain()
    assert treatment.primary.reasoning_effort == "low"
    assert treatment.primary._default_params["reasoning_effort"] == "low"
    assert base.primary.reasoning_effort == "high"
    assert base.primary._default_params["reasoning_effort"] == "high"
    monkeypatch.setattr(config_module, "FT_TREATMENT_REASONING_EFFORT", None)
    assert config_module._treatment_chain().primary.reasoning_effort is None


def test_treatment_instruction_only_on_treatment_arm():
    from types import SimpleNamespace

    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from langchain_core.messages import SystemMessage

    from src.middleware.model_experiment_middleware import CONCISE_INSTRUCTION

    m = lambda: FakeListChatModel(responses=["x"])  # noqa: E731
    exp = ModelExperiment(
        name="s",
        control=ModelChain(m(), (m(),)),
        treatment=ModelChain(m(), ()),
        treatment_percent=50,
        treatment_instruction=CONCISE_INSTRUCTION,
    )
    mw = ModelExperimentFallbackMiddleware(exp)

    class Req(SimpleNamespace):
        def override(self, **kw):
            return Req(**{**self.__dict__, **kw})

    sys_msg = SystemMessage(content="base prompt")
    out = mw._with_treatment_instruction(
        Req(state={"model_arm": "treatment"}, system_message=sys_msg)
    )
    texts = [b["text"] for b in out.system_message.content]
    assert texts == ["base prompt", CONCISE_INSTRUCTION]
    for arm in ("control", "base", None):
        req = Req(state={"model_arm": arm}, system_message=sys_msg)
        assert mw._with_treatment_instruction(req) is req
    off = ModelExperimentFallbackMiddleware(
        ModelExperiment(
            name="s",
            control=ModelChain(m(), (m(),)),
            treatment=None,
            treatment_percent=0,
            treatment_instruction=CONCISE_INSTRUCTION,
        )
    )
    req = Req(state={"model_arm": "treatment"}, system_message=sys_msg)
    assert off._with_treatment_instruction(req) is req


def test_treatment_reasoning_effort_defaults_to_high(config_module, monkeypatch):
    monkeypatch.delenv("FT_TREATMENT_REASONING_EFFORT", raising=False)
    assert (
        config_module._ft_reasoning_effort_from_env(
            "FT_TREATMENT_REASONING_EFFORT", "high"
        )
        == "high"
    )
