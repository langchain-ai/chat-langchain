from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.middleware.split_model_middleware import SplitModelMiddleware


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return "fact"


class _Fake(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_final_turn_goes_to_final_model():
    cheap = _Fake(messages=iter([
        AIMessage(content="", tool_calls=[{"name": "lookup", "args": {"q": "x"}, "id": "1"}]),
        AIMessage(content="cheap draft"),
    ]))
    strong = _Fake(messages=iter([AIMessage(content="strong answer")]))
    agent = create_agent(model=cheap, tools=[lookup], middleware=[SplitModelMiddleware(final_model=strong)])
    out = agent.invoke({"messages": [HumanMessage("hi")]})
    texts = [m.content for m in out["messages"] if isinstance(m, AIMessage)]
    assert texts[-1] == "strong answer"
    assert "cheap draft" not in texts
