from agent import docs_agent_middleware
from src.agent.config import model_fallback_middleware, model_retry_middleware
from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from src.middleware.latest_user_message_middleware import LatestUserMessageMiddleware


def test_latest_user_guard_precedes_retries_and_fallback():
    ingress_index = next(
        index
        for index, middleware in enumerate(docs_agent_middleware)
        if isinstance(middleware, IngressGuardsMiddleware)
    )
    latest_index = next(
        index
        for index, middleware in enumerate(docs_agent_middleware)
        if isinstance(middleware, LatestUserMessageMiddleware)
    )
    assert (
        ingress_index
        < latest_index
        < docs_agent_middleware.index(model_retry_middleware)
    )
    assert latest_index < docs_agent_middleware.index(model_fallback_middleware)


def test_regenerating_answer_guards_are_not_wired():
    # These guards re-invoke the model after a streamed answer, which surfaced
    # as a second (fallback-model) generation in the chat UI.
    regenerating = (
        AnswerSanityGuardMiddleware,
        CitationGuardMiddleware,
        DocsResearchGuardMiddleware,
    )
    assert not any(isinstance(m, regenerating) for m in docs_agent_middleware)
