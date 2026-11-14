from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from src.middleware.unused_tool_filter_middleware import UnusedToolFilterMiddleware


@tool
def write_file(path: str) -> str:
    """Write a file."""
    return "ok"


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return "found"


class RecordingModel(GenericFakeChatModel):
    bound: list = []

    def bind_tools(self, tools, **kwargs):
        RecordingModel.bound = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self


def test_unused_tools_are_not_bound():
    model = RecordingModel(messages=iter([AIMessage(content="done")]))
    agent = create_agent(
        model=model, tools=[write_file, lookup], middleware=[UnusedToolFilterMiddleware()]
    )
    agent.invoke({"messages": [HumanMessage(content="hi")]})
    assert RecordingModel.bound == ["lookup"]
