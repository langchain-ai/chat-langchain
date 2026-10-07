from agent import docs_agent_middleware
from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.ingress_guards_middleware import IngressGuardsMiddleware
from src.middleware.pending_turn_middleware import PendingTurnMiddleware


def test_pending_turn_guidance_follows_ingress_guards():
    assert isinstance(docs_agent_middleware[0], IngressGuardsMiddleware)
    assert isinstance(docs_agent_middleware[1], PendingTurnMiddleware)


def test_regenerating_answer_guards_are_not_wired():
    # These guards re-invoke the model after a streamed answer, which surfaced
    # as a second (fallback-model) generation in the chat UI.
    regenerating = (
        AnswerSanityGuardMiddleware,
        CitationGuardMiddleware,
        DocsResearchGuardMiddleware,
    )
    assert not any(isinstance(m, regenerating) for m in docs_agent_middleware)
