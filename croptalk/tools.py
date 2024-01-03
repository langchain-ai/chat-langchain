from langchain.tools import StructuredTool
from croptalk.retriever import (RetrieverInput, 
                                retriever_with_filter_by_category, 
                                retriever_with_filter_function)

doc_search_each_category = StructuredTool.from_function(
            name="doc_search_each_category",
            description="Searches and returns information in every doc category.",
            func=retriever_with_filter_by_category,
            args_schema=RetrieverInput,
        )

doc_search_any_category = StructuredTool.from_function(
            name="doc_search_any_category",
            description="Searches and returns information given the filters.",
            func=retriever_with_filter_function,
            args_schema=RetrieverInput,
        )


tools = [doc_search_each_category]