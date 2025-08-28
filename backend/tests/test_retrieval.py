
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
load_dotenv()

@dataclass
class QA:
    question: str
    criteria: str
    docs: list[str]


LANGGRAPH_QA_DATASET = [
    QA(
        question="""I was wondering how interrupt works when there is a variable of undeterminism. I have a llm that extracts python code from history. I added a random int and the end of the request, but it still proceeds with the run_python_code_tool without reprompting the user for hitl approval. Does this mean that if the llm extracts a different python code snippet, it will execute that one when the user had only approved the previous python snippet to be run
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
prompt = ChatPromptTemplate.from_template("Extract most recent python code "`)
response = llm.invoke(prompt.format(state=json.dumps(state)))
python_code = response.content
request = { code: python_code+str(random.randint(1, 10))}
response = interrupt(request)
if response["action"] = "accept":
      run_python_code_tool.invoke(python_code)""",
        criteria="Should discuss durable execution",
        docs=["https://docs.langchain.com/oss/python/durable-execution"]
    ),
    QA(
        question="Is there a prebuilt frontend I can use for chat that also supports interrupts?",
        criteria="Should point the user to Agent Chat UI",
        docs=["https://docs.langchain.com/oss/python/ui", "https://docs.langchain.com/oss/javascript/ui"]
    ),
    QA(
        question="Can I update the state of my graph myself",
        criteria="Yes, should have graph.update_state",
        docs=["https://docs.langchain.com/oss/python/5-customize-state", "https://docs.langchain.com/oss/javascript/5-customize-state"]
    ),
    QA(
        question="I want to debug my graph by restarting from previous steps. How can I do this?",
        criteria="Should discuss time travel and checkpointing",
        docs=["https://docs.langchain.com/oss/python/6-time-travel", "https://docs.langchain.com/oss/python/time-travel", "https://docs.langchain.com/langgraph-platform/langgraph-basics/6-time-travel"]
    )
]

LANGSMITH_QA_DATASET = [
    QA(
        question="""Is there any detailed doc about how to use RunTree? For example, how to create a new RunTree, how to append children, how to find a RunTree by ID, how to end a RunTree. I couldn’t find a way to get RunTree by ID, I can only get Run through client.readRun. How to convert this run to runtree? Thanks!""",
        criteria="Should discuss how to use RunTree",
        docs=["https://docs.langchain.com/langsmith/access-current-span", "https://docs.langchain.com/langsmith/annotate-code"]
    ),
    QA(
        question="""I am using the following code for tracing.
It shows the trace in the langsmith UI but it doesnt show the token usage, not the cost - how to fix that.

@traceable(run_type="llm", name='daniel1', metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-2025-04-14"})
def sample_tool()""",
        criteria="User needs to pass in prices to calculate token costs.",
        docs=["https://docs.langchain.com/langsmith/calculate-token-based-costs"]
    ),
    QA(
        question="Can I pass audio clips in while tracing?",
        criteria="Should say yes, pass as a file",
        docs=["https://docs.langchain.com/langsmith/upload-files-with-traces"]
    ),
    QA(
        question="I want to start a trace from my frontend, but I want that same trace to be appended to when my FE calls my BE endpoints. How can I do this?",
        criteria="Should discuss distributed tracing",
        docs=["https://docs.langchain.com/langsmith/distributed-tracing"]
    ),
    QA(
        question="If I have self-hosted langsmith and I share a trace publically, who can see it?",
        criteria="Should say that only people on your self-hosted instance can see it.",
        docs=["https://docs.langchain.com/langsmith/share-trace"]
    )
]

from backend.retrieval_graph.deepagent.deepagent import deep_agent
from backend.retrieval_graph.graph import graph
from langchain_core.messages import HumanMessage
from langsmith import Client

client = Client()

def create_langsmith_dataset():
    examples = []
    for qa in LANGGRAPH_QA_DATASET:
        examples.append(
            {
                "inputs": {
                    "messages": [HumanMessage(qa.question)]
                },
                "outputs": {
                    "criteria": qa.criteria,
                    "docs": qa.docs
                },
                "metadata": {
                    "subject": "langgraph"
                }
            }
        )
    for qa in LANGSMITH_QA_DATASET:
        examples.append(
            {
                "inputs": {
                    "messages": [HumanMessage(qa.question)]
                },
                "outputs": {
                    "criteria": qa.criteria,
                    "docs": qa.docs
                },
                "metadata": {
                    "subject": "langsmith"
                }
            }
        )
    client.create_examples(dataset_id="bb0a62ed-2d15-4953-ae22-0c65eb0e1063", examples=examples)


def found_docs(reference_outputs: dict, outputs: dict) -> bool:
    # Do the calculation on the documents
    # TODO: Maybe get rid of this
    if "documents" in outputs:
        for doc in outputs["documents"]:
            if doc.metadata["source"] in reference_outputs["docs"]:
                return True
        return False

    # We check that at least one of the docs is found in the response (typically between JS and Python)
    last_ai_message = None
    for msg in reversed(outputs["messages"]):
        if msg.type == "ai" and msg.text() != "":
            last_ai_message = msg
            break
    print("Last AI Message: ",last_ai_message)
    if not last_ai_message:
        return False
    for doc in reference_outputs["docs"]:
        if doc in last_ai_message.text():
            return True
    return False

async def target_func(inputs: dict):
    return await deep_agent.ainvoke(inputs)

async def original_clc_target_func(inputs: dict):
    return await graph.ainvoke(inputs)

if __name__ == "__main__":
    # create_langsmith_dataset()
    # async def main():
    #     await client.aevaluate(
    #         original_clc_target_func,
    #         data="bb0a62ed-2d15-4953-ae22-0c65eb0e1063",
    #         evaluators=[found_docs],
    #         experiment_prefix="original-clc: ",
    #         max_concurrency=10
    #     )
    # asyncio.run(main())

    ####################################
    # Create React Agent EXAMPLE
    ####################################
    from langchain_anthropic import ChatAnthropic
    from langgraph.prebuilt import create_react_agent
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver
    import requests

    @tool
    def say_zaijian(input: str) -> str:
        """
        Say zaijian to the user
        """
        return "Zaijian"

    @tool
    def say_adios(input: str) -> str:
        """
        Say adios to the user
        """
        return "Adios"

    @tool
    def say_goodbye(input: str) -> str:
        """
        Say goodbye to the user
        """
        return "Goodbye"

    from typing import Annotated
    from langchain_core.messages import BaseMessage
    from langgraph.graph import MessagesState
    from langgraph.graph.message import add_messages

    get_response = requests.get(
        "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md"
    )
    readme = get_response.text
    
    model = ChatAnthropic(model="claude-3-7-sonnet-20250219").bind(cache_control={"type": "ephemeral"})
    # model = ChatAnthropic(model="claude-3-7-sonnet-20250219")
    agent = create_react_agent(model, tools=[say_goodbye, say_adios, say_zaijian], checkpointer=MemorySaver())
    agent.invoke({"messages": [SystemMessage("Use every single tool to say goodbye, you don't know what language the user will respond best to, so it's best to use all of them. Call them in series! Not in parallel. Call say_goodbye first."), HumanMessage(f"{readme}")]}, config={"configurable": {"thread_id": "aaa"}})


    ####################################
    # Working EXAMPLE
    ####################################
    # import requests
    # from langchain_anthropic import ChatAnthropic
    # from langgraph.checkpoint.memory import MemorySaver
    # from langgraph.graph import START, StateGraph, add_messages
    # from typing_extensions import Annotated, TypedDict

    # model = ChatAnthropic(model="claude-3-7-sonnet-20250219").bind(cache_control={"type": "ephemeral"})

    # def messages_reducer(left: list, right: list) -> list:
    #     return add_messages(left, right)

    # class State(TypedDict):
    #     messages: Annotated[list, messages_reducer]

    # workflow = StateGraph(state_schema=State)

    # def call_model(state: State):
    #     response = model.invoke(state["messages"])
    #     return {"messages": [response]}
    # workflow.add_edge(START, "model")
    # workflow.add_node("model", call_model)
    # memory = MemorySaver()
    # app = workflow.compile(checkpointer=memory)

    # config = {"configurable": {"thread_id": "abc123"}}
    # # Question 1: Nothing
    # query = "Hi! I'm Bob."
    # input_message = HumanMessage([{"type": "text", "text": query}])
    # output = app.invoke({"messages": [input_message]}, config)
    #  # Question 2: Create cache
    # get_response = requests.get(
    #     "https://raw.githubusercontent.com/langchain-ai/langgraph/main/README.md"
    # )
    # readme = get_response.text
    # query = f"Check out this readme: {readme}"
    # input_message = HumanMessage([{"type": "text", "text": query}])
    # output = app.invoke({"messages": [input_message]}, config)
    # # Question 3: Use cache
    # query = "What was my name again?"
    # input_message = HumanMessage([{"type": "text", "text": query}])
    # output = app.invoke({"messages": [input_message]}, config)