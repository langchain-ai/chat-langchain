from langchain.agents.middleware import ModelFallbackMiddleware

from agent import docs_agent_middleware
from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.model_timeout_middleware import ModelTimeoutMiddleware


def test_regenerating_answer_guards_are_not_wired():
    # These guards re-invoke the model after a streamed answer, which surfaced
    # as a second (fallback-model) generation in the chat UI.
    regenerating = (
        AnswerSanityGuardMiddleware,
        CitationGuardMiddleware,
        DocsResearchGuardMiddleware,
    )
    assert not any(isinstance(m, regenerating) for m in docs_agent_middleware)


def test_model_timeout_wraps_fallback_attempts():
    fallback_index = next(
        index
        for index, middleware in enumerate(docs_agent_middleware)
        if isinstance(middleware, ModelFallbackMiddleware)
    )

    assert isinstance(docs_agent_middleware[fallback_index + 1], ModelTimeoutMiddleware)
