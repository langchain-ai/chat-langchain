# import matplotlib
# from langchain.tools import StructuredTool
#
# from croptalk.retriever import (RetrieverInput,
#                                 retriever_with_filter_each_category,
#                                 retriever_with_filter)
import os
from math import sqrt, cos, sin
from langchain.tools import tool
from typing import Optional, Union, Dict
import requests
from dotenv import load_dotenv

load_dotenv("secrets/.env.secret")


@tool("wfrp-commodities-tool")
def get_wfrp_commodities(reinsurance_yr: str, state_code: str, county_code: str) -> Optional[str]:
    """
    This tool is used to find commodities associated with the Whole Farm Revenue Protection (WFRP) insurance program
     for a given reinsurance year, state code and county code.

    To use this tool, you must provide all three arguments [reinsurance_ur, state_code and county_code]

    Args:
        reinsurance_yr (str): year of reinsurance, ex : "2024"
        state_code: (str) : state code, ex : "06" for California
        county_code: (str) : county code, ex : "001"

    Returns: Optional[str]
    """
    url = f"http://dev.lookup-api.cropguard.online/v1/lookups/GetWFRPCommodities"
    headers = {
        'accept': 'application/json',
        'api-key': os.environ["CROPGUARD_API_KEY"],
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, json={"ReinsuranceYear": reinsurance_yr,
                                                         "StateCode": state_code,
                                                         "CountyCode": county_code})

    if response.status_code == 200:
        print("API call successful")
        response_data = response.json()
        # Process the response data as needed
        return str([i["AGR Commodity Name"] for i in response_data])
    else:
        print("API call failed with status code:", response.status_code)
        return None

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


tools = [pythagoras_equation, get_wfrp_commodities]
