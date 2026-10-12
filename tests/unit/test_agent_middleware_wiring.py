from agent import docs_agent_middleware
from src.agent.config import model_fallback_middleware, model_retry_middleware
from src.middleware.answer_sanity_guard_middleware import AnswerSanityGuardMiddleware
from src.middleware.citation_guard_middleware import CitationGuardMiddleware
from src.middleware.docs_research_guard_middleware import DocsResearchGuardMiddleware
from src.middleware.tool_name_sanitizer_middleware import ToolNameSanitizerMiddleware


def test_tool_name_sanitizer_precedes_retry_and_fallback():
    retry_index = docs_agent_middleware.index(model_retry_middleware)
    assert isinstance(
        docs_agent_middleware[retry_index - 1], ToolNameSanitizerMiddleware
    )
    assert retry_index < docs_agent_middleware.index(model_fallback_middleware)


def test_regenerating_answer_guards_are_not_wired():
    # These guards re-invoke the model after a streamed answer, which surfaced
    # as a second (fallback-model) generation in the chat UI.
    regenerating = (
        AnswerSanityGuardMiddleware,
        CitationGuardMiddleware,
        DocsResearchGuardMiddleware,
    )
    assert not any(isinstance(m, regenerating) for m in docs_agent_middleware)
