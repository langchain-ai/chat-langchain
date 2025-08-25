

class QA():
    question: str
    answer: str
    docs: list[str]


LANGGRAPH_QA_DATASET = [
    QA(
        question="I have a langgraph agent deployed in production, and my TypeScript client needs to know whenever something goes wrong (e.g., HTTP error, API error, tool error). The idea is to show a toast notification in the UI and also keep developers informed in a clean, structured way. Up until now, my approach has been very simple: I keep a state variable like errorMessage, and whenever an exception happens I append the error text there. It works, but it feels a bit hacky and doesn’t scale well.",
        answer="Hi! Depending on the error, there are a few ways you can handle it. We have support for catching tool errors and turning them into messages. You can throw exceptions from your graph and catch them in your code that calls invoke on the graph. Your approach can work too if you don’t want to disrupt the graph’s execution, but if you want the graph to short circuit on an unexpected exception you probably want to raise an exception.",
        docs=["https://langchain-ai.github.io/langgraph/how-tos/tool-calling/#handle-errors"]
    ),
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
        answer="""When you resume a workflow run, the code does NOT resume from the same line of code where execution stopped; instead, it will identify an appropriate starting point from which to pick up where it left off. This means that the workflow will replay all steps from the starting point until it reaches the point where it was stopped.
As a result, when you are writing a workflow for durable execution, you must wrap any non-deterministic operations (e.g., random number generation) and any operations with side effects (e.g., file writes, API calls) inside tasks or nodes.
To ensure that your workflow is deterministic and can be consistently replayed, follow these guidelines:
Avoid Repeating Work: If a node contains multiple operations with side effects (e.g., logging, file writes, or network calls), wrap each operation in a separate task. This ensures that when the workflow is resumed, the operations are not repeated, and their results are retrieved from the persistence layer.
Encapsulate Non-Deterministic Operations: Wrap any code that might yield non-deterministic results (e.g., random number generation) inside tasks or nodes. This ensures that, upon resumption, the workflow follows the exact recorded sequence of steps with the same outcomes.
Use Idempotent Operations: When possible ensure that side effects (e.g., API calls, file writes) are idempotent. This means that if an operation is retried after a failure in the workflow, it will have the same effect as the first time it was executed. This is particularly important for operations that result in data writes. In the event that a task starts but fails to complete successfully, the workflow's resumption will re-run the task, relying on recorded outcomes to maintain consistency. Use idempotency keys or verify existing results to avoid unintended duplication, ensuring a smooth and predictable workflow execution.""",
        docs=["https://langchain-ai.github.io/langgraph/concepts/durable_execution/"]
    )
]

LANGSMITH_QA_DATASET = [
    QA(
        question="""Is there any detailed doc about how to use RunTree? For example, how to create a new RunTree, how to append children, how to find a RunTree by ID, how to end a RunTree. I couldn’t find a way to get RunTree by ID, I can only get Run through client.readRun. How to convert this run to runtree? Thanks!"""
        answer="""Hi! Depending on the error, there are a few ways you can handle it. We have support for catching tool errors and turning them into messages. You can throw exceptions from your graph and catch them in your code that calls invoke on the graph. Your approach can work too if you don’t want to disrupt the graph’s execution, but if you want the graph to short circuit on an unexpected exception you probably want to raise an exception.""",
        docs=["https://docs.smith.langchain.com/observability/how_to_guides/annotate_code#use-the-runtree-api", "https://docs.smith.langchain.com/reference/python/run_trees/langsmith.run_trees.RunTree#langsmith.run_trees.RunTree"]
    ),
    QA(
        question="""I am using the following code for tracing.
It shows the trace in the langsmith UI but it doesnt show the token usage, not the cost - how to fix that.

@traceable(run_type="llm", name='daniel1', metadata={"ls_provider": "openai", "ls_model_name": "gpt-4.1-2025-04-14"})
def sample_tool()"""
        answer="""you can use this model pricing table to calculate token cost """,
        docs=["https://docs.smith.langchain.com/observability/how_to_guides/calculate_token_based_costs"]
    )
]

# TODO: Set up a test for the retrieval graph that runs with latest assistant

