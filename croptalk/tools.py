# import matplotlib
# from langchain.tools import StructuredTool
#
# from croptalk.retriever import (RetrieverInput,
#                                 retriever_with_filter_each_category,
#                                 retriever_with_filter)

from math import sqrt, cos, sin
from langchain.tools import tool
from typing import Optional, Union

# doc_search_each_category = StructuredTool.from_function(
#     name="doc_search_each_category",
#     description="Searches and returns information in every doc category.",
#     func=retriever_with_filter_each_category,
#     args_schema=RetrieverInput,
# )
#
# doc_search_any_category = StructuredTool.from_function(
#     name="doc_search_any_category",
#     description="Searches and returns information given the filters. Disregards the doc category.",
#     func=retriever_with_filter,
#     args_schema=RetrieverInput,
# )


@tool("pythagoras-equation-tool")
def pythagoras_equation(adjacent_side: Optional[Union[int, float]] = None,
                        opposite_side: Optional[Union[int, float]] = None,
                        angle: Optional[Union[int, float]] = None
                        ):
    """
    use this tool when you need to calculate the length of an hypotenuse
    given one or two sides of a triangle and/or an angle (in degrees).
    To use the tool you must provide at least two of the following parameters
    "['adjacent_side', 'opposite_side', 'angle']."
    """

    # check for the values we have been given
    if adjacent_side and opposite_side:
        return sqrt(float(adjacent_side) ** 2 + float(opposite_side) ** 2)
    elif adjacent_side and angle:
        return adjacent_side / cos(float(angle))
    elif opposite_side and angle:
        return opposite_side / sin(float(angle))
    else:
        return "Could not calculate the hypotenuse of the triangle. Need two or more of `adjacent_side`, `opposite_side`, or `angle`."


tools = [pythagoras_equation]
