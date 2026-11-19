"""Thread-pinned A/B split between the default model and a post-trained model.

Each new thread is assigned an arm from a salted sha256 of its ``thread_id``.
The arm is saved in private agent state on the thread's first turn, so a
conversation keeps its arm even if the traffic percentage or salt changes
later. Threads that already had answers before the experiment existed are
pinned to control.

The saved arm is the bare word ``treatment`` or ``control``: it does not
record which model or experiment it came from. So when a new LoRA is loaded
into ``FT_MODEL_ID``, treatment threads that are already open continue on the
new model. For a clean break between model versions, also change
``FT_MODEL_EXPERIMENT`` in ``src/agent/config.py``. That only re-salts new
threads, though: old treatment threads still run whatever ``FT_MODEL_ID``
points to. The ``model_experiment`` label comes from the model, not the
thread, so an old thread's later turns carry the new label; only its earlier
turns keep the old one. Spot these threads in LangSmith as treatment threads
with runs under both labels. Saving the experiment name with the arm would
close this gap; it was left out on purpose to keep the pin simple.

Two middlewares share one :class:`ModelExperiment`:

* :class:`ModelArmPinMiddleware` goes first in the middleware list, so it
  pins the arm before any other ``before_agent`` hook can end the turn early.
* :class:`ModelExperimentFallbackMiddleware` replaces the plain fallback
  middleware and runs the pinned arm's primary model and fallback chain.

Every model in both chains carries ``model_experiment``, ``model_arm`` and
``model_role`` (``primary`` / ``fallback``) metadata, which LangSmith puts on
each model-call run next to the ``thread_id`` LangGraph already adds. Query
``model_arm = treatment AND model_role = fallback`` to find treatment threads
that were not answered by the post-trained model.

Model-call metadata only reaches the LLM child runs, and metadata set on the
root run from inside the graph doesn't persist in production, so while the
experiment is on every turn's trace also gets feedback that eval rules and
root filters can match: ``model_arm_treatment`` (score 1 for treatment, 0 for
control), ``model_fallback`` (score 1 when a primary model failed and the
turn fell back), and ``model_arm`` / ``model_experiment`` as string values
for reading. The same keys are also set on the root run's metadata where the
SDK allows it.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Annotated, Any

import langsmith as ls
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    AgentState,
    ModelRequest,
    PrivateStateAttr,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from typing_extensions import NotRequired

from src.middleware.model_fallback_middleware import RepairingModelFallbackMiddleware

logger = logging.getLogger(__name__)

ARM_CONTROL = "control"
ARM_TREATMENT = "treatment"
ARMS = frozenset({ARM_CONTROL, ARM_TREATMENT})


def assign_arm(thread_id: str, salt: str, treatment_percent: int) -> str:
    """Return the arm for ``thread_id``; the same inputs always give the same arm.

    Uses sha256 rather than ``hash()``, which is randomized per process.
    """
    digest = hashlib.sha256(f"{salt}:{thread_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return ARM_TREATMENT if bucket < treatment_percent else ARM_CONTROL


@dataclass(frozen=True)
class ModelChain:
    """A primary model and the fallbacks tried, in order, when it fails."""

    primary: BaseChatModel
    fallbacks: tuple[BaseChatModel, ...]


@dataclass(frozen=True)
class ModelExperiment:
    """Experiment config shared by the pin and fallback middlewares.

    ``treatment=None`` turns the experiment off: every thread runs the control
    chain, including threads already pinned to treatment (the kill switch).
    ``treatment_percent=0`` with a treatment chain stops new enrollment but
    keeps already-pinned treatment threads on the post-trained model.
    """

    name: str
    control: ModelChain
    treatment: ModelChain | None
    treatment_percent: int

    def __post_init__(self) -> None:
        """Reject a traffic percentage outside 0-100."""
        if not 0 <= self.treatment_percent <= 100:
            raise ValueError("treatment_percent must be between 0 and 100")

    @property
    def enabled(self) -> bool:
        """Return whether a treatment model is configured."""
        return self.treatment is not None

    def chain_for(self, arm: object) -> ModelChain:
        """Return the model chain for a pinned arm; anything else gets control."""
        if self.treatment is not None and arm == ARM_TREATMENT:
            return self.treatment
        return self.control


class ModelExperimentState(AgentState):
    """Agent state with the pinned arm. Hidden from run input and output."""

    model_arm: NotRequired[Annotated[str, PrivateStateAttr]]


def _tag_root_run(
    metadata: dict[str, str | bool], scores: dict[str, int] | None = None
) -> None:
    """Log ``metadata`` and ``scores`` as feedback on the current trace.

    Never fails the turn. Passing ``trace_id`` batches the feedback in the
    background instead of blocking on a request.
    """
    try:
        run_tree = ls.get_current_run_tree()
        if not run_tree:
            return
        while run_tree.parent_run is not None:
            run_tree = run_tree.parent_run
        run_tree.metadata.update(metadata)
        client, trace_id = run_tree.client, run_tree.trace_id
        for key, value in metadata.items():
            if isinstance(value, bool):
                client.create_feedback(trace_id=trace_id, key=key, score=int(value))
            else:
                client.create_feedback(trace_id=trace_id, key=key, value=value)
        for key, score in (scores or {}).items():
            client.create_feedback(trace_id=trace_id, key=key, score=score)
    except Exception:
        logger.debug("Could not tag the root run with %s", metadata, exc_info=True)


def _current_thread_id() -> str | None:
    from langgraph.config import get_config

    try:
        configurable = get_config().get("configurable") or {}
    except RuntimeError:  # outside a runnable context
        return None
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else None


class ModelArmPinMiddleware(AgentMiddleware[ModelExperimentState]):
    """Pin the thread's model arm on its first turn and tag every turn's root run."""

    state_schema = ModelExperimentState

    def __init__(self, experiment: ModelExperiment) -> None:
        """Pin arms for ``experiment``."""
        super().__init__()
        self.experiment = experiment

    def _initial_arm(self, state: dict[str, Any]) -> str:
        if not self.experiment.enabled:
            return ARM_CONTROL
        # Threads that were already answering before the experiment started
        # stay on the model they began with.
        if any(isinstance(m, AIMessage) for m in state.get("messages") or ()):
            return ARM_CONTROL
        thread_id = _current_thread_id()
        if not thread_id:
            return ARM_CONTROL
        return assign_arm(
            thread_id, self.experiment.name, self.experiment.treatment_percent
        )

    def before_agent(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        """Save the arm on the thread's first turn and label the turn's root run."""
        arm = state.get("model_arm")
        update = None
        if arm not in ARMS:
            arm = self._initial_arm(state)
            update = {"model_arm": arm}
        if self.experiment.enabled:
            _tag_root_run(
                {"model_experiment": self.experiment.name, "model_arm": arm},
                {"model_arm_treatment": int(arm == ARM_TREATMENT)},
            )
        return update

    async def abefore_agent(
        self, state: dict[str, Any], runtime: Any
    ) -> dict[str, Any] | None:
        """Async version of :meth:`before_agent`."""
        return self.before_agent(state, runtime)


class ModelExperimentFallbackMiddleware(RepairingModelFallbackMiddleware):
    """Fallback middleware that runs the pinned arm's model chain.

    ``primary_model`` and ``models`` stay the control chain, so code that
    inspects them sees the default behavior.
    """

    state_schema = ModelExperimentState

    def __init__(self, experiment: ModelExperiment) -> None:
        """Run the control or treatment chain of ``experiment`` per thread."""
        super().__init__(
            *experiment.control.fallbacks, primary_model=experiment.control.primary
        )
        self.experiment = experiment

    def _model_chain(
        self, request: ModelRequest
    ) -> tuple[BaseChatModel | None, list[BaseChatModel]]:
        chain = self.experiment.chain_for((request.state or {}).get("model_arm"))
        return chain.primary, list(chain.fallbacks)

    def _on_fallback(self, request: ModelRequest) -> None:
        if self.experiment.enabled:
            _tag_root_run({"model_fallback": True})


__all__ = [
    "ARM_CONTROL",
    "ARM_TREATMENT",
    "ModelArmPinMiddleware",
    "ModelChain",
    "ModelExperiment",
    "ModelExperimentFallbackMiddleware",
    "ModelExperimentState",
    "assign_arm",
]
