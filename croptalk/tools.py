from langchain.tools import StructuredTool
from croptalk.retriever import (RetrieverInput,
                                retriever_with_filter_each_category,
                                retriever_with_filter)

doc_search_each_category = StructuredTool.from_function(
    name="doc_search_each_category",
    description="Searches and returns information in every doc category.",
    func=retriever_with_filter_each_category,
    args_schema=RetrieverInput,
)

doc_search_any_category = StructuredTool.from_function(
    name="doc_search_any_category",
    description="Searches and returns information given the filters. Disregards the doc category.",
    func=retriever_with_filter,
    args_schema=RetrieverInput,
)


tools = [doc_search_each_category]
