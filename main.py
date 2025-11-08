from dotenv import load_dotenv

load_dotenv()

from backend.retrieval_graph.graph import graph

# if __name__ == "__main__":
#     print("Hello Advanced RAG")
#     # print(graph.invoke(input={"question": "LLM Token Manipulation"}))
#     # print(graph.invoke(input={"question": "How to use MCP in LangChain?"}))
#     print(graph.invoke(input={"question": "How to use MCP in LangChain?"}))


if __name__ == "__main__":
    import asyncio
    import warnings
    from langchain_core.messages import HumanMessage

    # Suppress ResourceWarning for unclosed sockets (common with async HTTP clients)
    # These warnings occur because LangChain's HTTP clients aren't explicitly closed,
    # but they're cleaned up by Python's garbage collector, so they're not critical.
    warnings.filterwarnings(
        "ignore", category=ResourceWarning, message=".*unclosed.*socket.*"
    )

    async def test_graph():
        try:
            result = await graph.ainvoke(
                input={
                    "messages": [HumanMessage(content="What is langchaun-mcp-adapter?")]
                }
            )
            print("Graph execution completed!")
            print(f"Answer: {result.get('answer', 'N/A')}")
            print(f"Steps: {result.get('steps', [])}")
            print(f"Documents retrieved: {len(result.get('documents', []))}")
        finally:
            # Give async tasks time to clean up connections
            await asyncio.sleep(0.1)

    asyncio.run(test_graph())
