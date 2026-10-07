from agent import docs_agent_middleware
from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from src.middleware.orphan_turn_middleware import OrphanHumanMessageMiddleware


def test_orphan_turn_middleware_follows_ingress_guards():
    ingress_index = next(
        index
        for index, middleware in enumerate(docs_agent_middleware)
        if isinstance(middleware, IngressGuardsMiddleware)
    )
    assert isinstance(
        docs_agent_middleware[ingress_index + 1], OrphanHumanMessageMiddleware
    )


def test_regenerating_answer_guards_are_not_wired():
    # These guards re-invoke the model after a streamed answer, which surfaced
    # as a second (fallback-model) generation in the chat UI.
    regenerating = (
        AnswerSanityGuardMiddleware,
        CitationGuardMiddleware,
        DocsResearchGuardMiddleware,
    )
    assert not any(isinstance(m, regenerating) for m in docs_agent_middleware)
