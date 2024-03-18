import os
from math import sqrt, cos, sin
from typing import Optional, Union

import requests
from dotenv import load_dotenv
from langchain.tools import tool

from croptalk.load_data import SOB

load_dotenv("secrets/.env.secret")


@tool("get-SOB-metrics")
def get_sob_metrics_for_crop_county(state_abbreviation: str, county_name: str, commodity_name: str,
                                    insurance_plan_name: str, metric: str) -> str:
    """
    This tool is used to query the summary of business data (SOB) to retrieve insurance program metrics by coverage level.

    Those metric can be percentage liability indemnified (pct_liability_indemnified), cost to grower (cost_to_grower),
    the number of policies sold (policies_sold_count) or the premium per quantity (premium_per_quantity)

    For instance, a user might ask : "What is the percentage of policies indemnified for Pierce county in North Dakota,
    for sunflowers under the APH program?". The tool will answer this question with the following arguments :
    (if arguments are not provided, they should be set to "None")

    Args:
        state_abbreviation: two letter string of State abbreviation (California -> CA, North Dakota -> ND and so on)
        county_name: name of county provided
        commodity_name: name of commodity
        insurance_plan_name: provided plan name abbreviation
        metric : choice from ["pct_liability_indemnified", "cost_to_grower", "policies_sold_count", "premium_per_quantity"]

    Returns: the tool returns a string which gives the coverage level with their
            associated percentage indemnified statistic

    """

    sob = SOB[SOB["commodity_year"] == 2023]

    if state_abbreviation == "":
        state_abbreviation = "None"
    if county_name == "":
        county_name = "None"
    if insurance_plan_name == "":
        insurance_plan_name = "None"
    if commodity_name == "":
        commodity_name = "None"

    sob = sob[
        (sob["state_abbreviation"] == state_abbreviation) &
        (sob["county_name"] == county_name) &
        (sob["commodity_name"] == commodity_name.lower()) &
        (sob["insurance_plan_name_abbreviation"] == insurance_plan_name)
        ]

    if not sob.empty:
        response = "In 2023, we observe the following data across coverage level : "
        for i, j in zip(list(sob["coverage_level"]), list(sob[metric])):
            response += f" coverage level : {i}, {metric} : {j} \b"

        if metric == "policies_sold_count":
            response += f"Total : {sob[metric].sum()}"
            # TODO else : weighted mean
        return response

    return f"MISSING_DATA : The requested metric ({metric}) for state : {state_abbreviation}, " \
           f" county : {county_name}, commodity : {commodity_name} and insurance plan : {insurance_plan_name}, " \
           f"is not available within the summary of business data. Make sure that you are providing an " \
           "existing combination of county, commodity and insurance plan name."


@tool("wfrp-commodities-tool")
def get_wfrp_commodities(reinsurance_yr: str, state_code: str, county_code: str) -> Optional[str]:
    """
    This tool is used to find commodities associated with the Whole Farm Revenue Protection (WFRP) insurance program
    for a given reinsurance year, state code and county code. Do not use this tool unless you are explicitly asked
    about available commodities for the WFRP program.

    To use this tool, you must provide all three arguments [reinsurance_yr, state_code and county_code]
    Do not make up those arguments if they are not provided explicitly.

    Args:
        reinsurance_yr (str): year of reinsurance
        state_code: (str) : state code
        county_code: (str) : county code

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
        commodity_list = [i["AGR Commodity Name"] for i in response_data]
        return f"Available commodities for the WFRP program (year {reinsurance_yr}, state_code {state_code}, and " \
               f"county_code {county_code}) are : " + str(commodity_list)

    else:
        print("API call failed with status code:", response.status_code)
        return None


tools = [get_wfrp_commodities, get_sob_metrics_for_crop_county]
