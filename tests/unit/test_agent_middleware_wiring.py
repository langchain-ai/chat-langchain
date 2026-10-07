from agent import docs_agent_middleware
from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.guardrails_middleware import GuardrailsMiddleware
from src.middleware.stale_turn_middleware import StaleTurnMiddleware
from src.middleware.summarization_middleware import CustomSummarizationMiddleware


def test_stale_turn_middleware_runs_after_guardrails_before_summarization():
    middleware_types = [type(middleware) for middleware in docs_agent_middleware]

    guardrails_index = middleware_types.index(GuardrailsMiddleware)
    assert middleware_types[guardrails_index + 1] is StaleTurnMiddleware
    assert middleware_types[guardrails_index + 2] is CustomSummarizationMiddleware


def test_regenerating_answer_guards_are_not_wired():
    # These guards re-invoke the model after a streamed answer, which surfaced
    # as a second (fallback-model) generation in the chat UI.
    regenerating = (
        AnswerSanityGuardMiddleware,
        CitationGuardMiddleware,
        DocsResearchGuardMiddleware,
    )
    assert not any(isinstance(m, regenerating) for m in docs_agent_middleware)
