# query_weaviate.py
import argparse
from langchain_core.runnables import RunnableConfig
from backend.retrieval import make_retriever


def main():
    parser = argparse.ArgumentParser(description="Query Weaviate vector store")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "-k", type=int, default=5, help="Number of results (default: 5)"
    )
    parser.add_argument(
        "--show-content", action="store_true", help="Show document content"
    )
    args = parser.parse_args()

    config = RunnableConfig(configurable={"search_kwargs": {"k": args.k}})

    with make_retriever(config) as retriever:
        docs = retriever.invoke(args.query)
        print(f"\nFound {len(docs)} documents for: '{args.query}'\n")

        for i, doc in enumerate(docs, 1):
            print(f"{i}. {doc.metadata['title']}")
            print(f"   Source: {doc.metadata['source'][:70]}")
            print(f"   Distance: {doc.metadata['distance']:.4f}")
            if args.show_content:
                # print(f"   Content: {doc.page_content[:200]}...")
                print(f"   Content: {doc.page_content}")
            print()


if __name__ == "__main__":
    main()


# # Basic query
# PYTHONPATH=. poetry run python query_weaviate.py "What is LangChain?"

# # With custom k value
# PYTHONPATH=. poetry run python query_weaviate.py "How do I use agents?" -k 3

# # Show content preview
# PYTHONPATH=. poetry run python query_weaviate.py "What are chains?" --show-content
