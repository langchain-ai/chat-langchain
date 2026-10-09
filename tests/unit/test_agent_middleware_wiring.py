from agent import docs_agent_middleware
from src.agent.config import duplicate_call_guard_middleware
from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.search_result_trim_middleware import SearchResultTrimMiddleware


def test_search_trimming_follows_duplicate_call_guard():
    index = docs_agent_middleware.index(duplicate_call_guard_middleware)
    assert isinstance(docs_agent_middleware[index + 1], SearchResultTrimMiddleware)


def test_regenerating_answer_guards_are_not_wired():
    # These guards re-invoke the model after a streamed answer, which surfaced
    # as a second (fallback-model) generation in the chat UI.
    regenerating = (
        AnswerSanityGuardMiddleware,
        CitationGuardMiddleware,
        DocsResearchGuardMiddleware,
    )
    assert not any(isinstance(m, regenerating) for m in docs_agent_middleware)
